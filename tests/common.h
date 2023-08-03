#pragma once
#include "bark.h"

#include <tuple>
#include <vector>

#define ABS_TOL 0.01f
#define REL_TOL 0.01f

typedef std::vector<float> logit_sequence;
typedef std::tuple<bark_sequence, logit_sequence, logit_sequence> test_data_t;

bool run_test_on_sequence(logit_sequence truth, logit_sequence logits);

void load_test_data(std::string fname, std::vector<int>& input, std::vector<float>& logits);