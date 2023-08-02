#pragma once
#include "bark.h"

#include <tuple>
#include <vector>

#define TOL 0.05f

typedef std::vector<float> logit_sequence;
typedef std::tuple<bark_sequence, logit_sequence, logit_sequence> test_data_t;

bool run_test(logit_sequence truth, logit_sequence logits, bool merge_ctx);