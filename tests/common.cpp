#include <fstream>
#include <vector>
#include <tuple>

#include "bark-util.h"
#include "common.h"

int64_t bytes_left(std::ifstream & f) {
    // utils to check all bytes are read from stream
    int64_t curr_pos = f.tellg();
    f.seekg(0, std::ios::end);
    int64_t file_size = f.tellg();
    int64_t bytes_left_to_read = file_size - curr_pos;
    return bytes_left_to_read;
}

template <typename T, typename U>
inline bool all_close(
            std::vector<T>   s1,
            std::vector<U>   s2,
                     float * max_violation,
                       int * n_violations) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        float violation = fabs(s1[i] - s2[i]);
        *max_violation = std::max(*max_violation, violation);
        if (violation > ABS_TOL)
            *n_violations += 1;
    }
    return *n_violations == 0;
}

template bool all_close(
                std::vector<int>   s1,
                std::vector<int>   s2,
                           float * max_violation,
                             int * n_violations);

template bool all_close(
                std::vector<float>   s1,
                std::vector<float>   s2,
                             float * max_violation,
                               int * n_violations);

inline bool all_close(
    std::vector<std::vector<float>>   s1,
    std::vector<std::vector<float>>   s2,
                              float * max_violation,
                                int * n_violations) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        if (s1[i].size() != s2[i].size()) { return false; }
        for (int j = 0; j < (int) s1[i].size(); j++) {
            float violation = fabs(s1[i][j] - s2[i][j]);
            *max_violation = std::max(*max_violation, violation);
            if (violation > ABS_TOL)
                *n_violations += 1;
        }
    }
    return *n_violations == 0;
}

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
inline bool all_equal(
            std::vector<std::vector<T>>   s1,
            std::vector<std::vector<U>>   s2,
                                    int * n_violations) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        if (s1[i].size() != s2[i].size()) { return false; }
        for (int j = 0; j < (int) s1[i].size(); j++) {
            if (s1[i][j] != s2[i][j])
                *n_violations += 1;
        }
    }
    return *n_violations == 0;
}

template bool all_equal(
            std::vector<std::vector<int>>,
            std::vector<std::vector<int>>,
            int * n_violations);

bool run_test(std::vector<float> truth, std::vector<float> result) {
    float max_violation = 0.0f;
    int n_violations = 0;
    if (!all_close(result, truth, &max_violation, &n_violations)) {
        if (n_violations == 0) {
            fprintf(stderr, "%s : wrong shape (%zu != %zu).\n", __func__, truth.size(), result.size());
        } else {
            fprintf(stderr, "%s: abs_tol=%.4f, abs max viol=%.4f, viol=%.1f%%", __func__, ABS_TOL, max_violation, (float)n_violations/truth.size()*100);
            fprintf(stderr, "\n");
        }
        return false;
    }
    return true;
}

bool run_test(std::vector<int> truth, std::vector<int> result) {
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

bool run_test(logit_matrix truth, logit_matrix result) {
    float max_violation = 0.0f;
    int n_violations = 0;
    if (!all_close(result, truth, &max_violation, &n_violations)) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s : failed test\n", __func__);
        if (n_violations == 0) {
            fprintf(stderr, "%s : wrong shape (%zu != %zu).\n", __func__, truth.size(), result.size());
        } else {
            fprintf(stderr, "       abs_tol=%.4f, abs max viol=%.4f, viol=%.1f%%", ABS_TOL, max_violation, (float)n_violations/(truth.size()*truth[0].size())*100);
            fprintf(stderr, "\n");
        }
        return false;
    }
    return true;
}

bool run_test(bark_codes truth, bark_codes result) {
    int n_violations = 0;
    if (!all_equal(result, truth, &n_violations)) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s : failed test\n", __func__);
        if (n_violations == 0) {
            fprintf(stderr, "%s : wrong shape (%zu != %zu).\n", __func__, truth.size(), result.size());
        } else {
            fprintf(stderr, "       viol=%.1f%%", (float)n_violations/(truth.size()*truth[0].size())*100);
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

    if (bytes_left(fin) > 0) {
        throw std::runtime_error("EOF not reached");
    }
}

template void load_test_data(
                 std::string   fname,
        std::vector<int32_t> & input,
          std::vector<float> & output);

template void load_test_data(
                 std::string   fname,
        std::vector<int32_t> & input,
        std::vector<int32_t> & output);

template <typename T, typename U>
void load_test_data(
                        std::string   fname,
                     std::vector<T> & input,
        std::vector<std::vector<U>> & output) {
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

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; i++) { read_safe(fin, ne[n_dims-1-i]); }

        for (int i = 0; i < ne[0]; i++) {
            std::vector<U> _tmp(ne[1]);
            fin.read(reinterpret_cast<char *>(_tmp.data()), ne[1]*sizeof(U));
            output.push_back(_tmp);
        }
    }

    if (bytes_left(fin) > 0) {
        throw std::runtime_error("EOF not reached");
    }
}

template void load_test_data(std::string fname, std::vector<int32_t>& input, std::vector<std::vector<int32_t>>& output);

void load_test_data(std::string fname, std::vector<std::vector<int32_t>>& input, std::vector<float>& output) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        throw;
    }

    // input
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; i++) { read_safe(fin, ne[n_dims-1-i]); }

        for (int i = 0; i < ne[0]; i++) {
            std::vector<int32_t> _tmp(ne[1]);
            fin.read(reinterpret_cast<char *>(_tmp.data()), ne[1]*sizeof(int32_t));
            input.push_back(_tmp);
        }
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
        fin.read(reinterpret_cast<char *>(output.data()), nelements*sizeof(float));
    }

    if (bytes_left(fin) > 0) {
        throw std::runtime_error("EOF not reached");
    }
}

template <typename T>
void load_test_data(
        std::string fname,
        std::vector<std::vector<int32_t>> & input,
        std::vector<std::vector<T>>       & output) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        throw;
    }

    // input
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; i++) { read_safe(fin, ne[n_dims-i-1]); }

        for (int i = 0; i < ne[0]; i++) {
            std::vector<int> _tmp(ne[1]);
            fin.read(reinterpret_cast<char *>(_tmp.data()), ne[1]*sizeof(int32_t));
            input.push_back(_tmp);
        }
    }

    // output
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t ne[3] = { 1, 1, 1 };
        for (int i = 0; i < n_dims; i++) { read_safe(fin, ne[n_dims-i-1]); }

        for (int i = 0; i < ne[0]; i++) {
            std::vector<T> _tmp(ne[1]);
            fin.read(reinterpret_cast<char *>(_tmp.data()), ne[1]*sizeof(T));
            output.push_back(_tmp);
        }
    }

    if (bytes_left(fin) > 0) {
        throw std::runtime_error("EOF not reached");
    }
}

template void load_test_data(
                    std::string fname,
                    std::vector<std::vector<int32_t>> & input,
                    std::vector<std::vector<float>>   & output);

template void load_test_data(
                    std::string fname,
                    std::vector<std::vector<int32_t>> & input,
                    std::vector<std::vector<int32_t>> & output);
