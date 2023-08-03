#pragma once
#include "bark.h"

#include <tuple>
#include <vector>

#define ABS_TOL 0.01f
#define REL_TOL 0.01f

typedef std::vector<float> logit_sequence;
typedef std::vector<std::vector<float>> logit_matrix;

bool run_test_on_sequence(logit_sequence truth, logit_sequence logits);

bool run_test_on_codes(logit_matrix truth, logit_matrix logits);

void load_test_data(
        std::string fname,
        std::vector<int>& input,
        logit_sequence& logits);

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
