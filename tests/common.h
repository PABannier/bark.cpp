#pragma once
#include "bark.h"

#include <tuple>
#include <vector>

#define ABS_TOL 0.01f
#define REL_TOL 0.01f

typedef std::vector<float> logit_sequence;
typedef std::vector<std::vector<float>> logit_matrix;

bool run_test_on_sequence(std::vector<int> truth, std::vector<int> result);
bool run_test_on_sequence(std::vector<float> truth, std::vector<float> result);

bool run_test_on_codes(logit_matrix truth, logit_matrix result);
bool run_test_on_codes(bark_codes truth, bark_codes result);

template <typename T, typename U>
void load_test_data(std::string fname, std::vector<T>& input, std::vector<U>& output);

template <typename T, typename U>
void load_test_data(std::string fname, std::vector<T>& input, std::vector<std::vector<U>>& output);

void load_nested_test_data(
        std::string fname,
        std::vector<std::vector<int>>& input,
        logit_matrix& logits);

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
