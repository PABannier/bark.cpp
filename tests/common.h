#pragma once
#include "bark.h"

#include <tuple>
#include <vector>

#define ABS_TOL 0.001f

typedef std::vector<float> logit_sequence;
typedef std::vector<float> audio_arr_t;
typedef std::vector<std::vector<float>> logit_matrix;
typedef std::vector<std::vector<int32_t>> bark_codes;

/* Comparison utils */
template <typename T, typename U>
inline bool all_equal(std::vector<T> s1, std::vector<U> s2, int * n_violations);

template <typename T, typename U>
inline bool all_equal(
            std::vector<std::vector<T>>   s1,
            std::vector<std::vector<U>>   s2,
                                    int * n_violations);

template <typename T, typename U>
inline bool all_close(
            std::vector<T>   s1,
            std::vector<U>   s2,
                     float * max_violation,
                       int * n_violations);

/* Test utils */
bool run_test(std::vector<int> truth, std::vector<int> result);
bool run_test(std::vector<float> truth, std::vector<float> result);

bool run_test(logit_matrix truth, logit_matrix result);
bool run_test(bark_codes truth, bark_codes result);

/* Load utils */
template <typename T, typename U>
void load_test_data(std::string fname, std::vector<T>& input, std::vector<U>& output);

template <typename T, typename U>
void load_test_data(std::string fname, std::vector<T>& input, std::vector<std::vector<U>>& output);

void load_test_data(std::string fname, std::vector<std::vector<int32_t>>& input, std::vector<float>& output);

template <typename T>
void load_test_data(
                std::string fname,
                std::vector<std::vector<int32_t>> & input,
                std::vector<std::vector<T>>       & output);

template <typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    std::vector<std::vector<T>> result(data[0].size(), std::vector<T>(data.size()));
    for (size_t i = 0; i < data[0].size(); i++)
        for (size_t j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}
