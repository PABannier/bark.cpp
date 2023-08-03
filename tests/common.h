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
