#pragma once
#include "encodec.h"

#include <map>
#include <random>
#include <vector>

#define SAMPLE_RATE 24000

#define CLS_TOKEN_ID 101
#define SEP_TOKEN_ID 102

#define TEXT_ENCODING_OFFSET 10048
#define TEXT_PAD_TOKEN 129595

#define CODEBOOK_SIZE 1024
#define N_COARSE_CODEBOOKS 2
#define N_FINE_CODEBOOKS 8

#define SEMANTIC_PAD_TOKEN 10000
#define SEMANTIC_INFER_TOKEN 129599
#define SEMANTIC_VOCAB_SIZE 10000
#define SEMANTIC_RATE_HZ 49.9

#define COARSE_RATE_HZ 75
#define COARSE_SEMANTIC_PAD_TOKEN 12048
#define COARSE_INFER_TOKEN 12050

struct bark_context;

struct bark_perf_stats;

struct bark_context_params {
    uint32_t seed;         // RNG seed, -1 for random

    float temp;
    float temp_fine;

    float min_eos_p;

    int sliding_window_size;
    int max_coarse_history;
};

struct gpt_hparams {
    int32_t n_in_vocab;
    int32_t n_out_vocab;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_embd;
    int32_t block_size;
    int32_t n_lm_heads;
    int32_t n_wtes;

    int32_t n_codes_given = 1;
};

struct bark_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    std::map<token, id> subword_token_to_id;
    std::map<id, token> id_to_subword_token;
};

typedef std::vector<bark_vocab::id>              bark_sequence;
typedef std::vector<std::vector<bark_vocab::id>> bark_codes;
typedef std::vector<float>                       audio_arr_t;

struct gpt_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // mlp
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt_model {
    gpt_hparams hparams;

    struct ggml_context * ctx = NULL;

    std::vector<float> logits;
    struct ggml_tensor * output_tokens;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wpe;

    std::vector<struct ggml_tensor *> wtes;
    std::vector<struct ggml_tensor *> lm_heads;

    std::vector<gpt_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    std::map<std::string, struct ggml_tensor *> tensors;

    ~gpt_model() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};


struct bark_model {
    // encoder
    gpt_model & text_model;
    gpt_model & coarse_model;
    gpt_model & fine_model;

    // decoder
    encodec_model & codec_model;

    bark_vocab vocab;

    float eos_p;  // for semantic model only

    ~bark_model() {
        delete &coarse_model, &fine_model, &text_model, &codec_model;
    }
};

bool gpt_model_load(const std::string& fname, gpt_model& model);

bool gpt_eval(
        const gpt_model & model,
        const int n_threads,
        int * n_past,
        const bool merge_ctx,
        const bark_sequence & embd_inp,
              std::vector<float>          & embd_w,
              size_t                      & mem_per_token);

bool fine_gpt_eval(
        const gpt_model & model,
        const int n_threads,
        const int codebook_ix,
        const bark_codes & embd_inp,
              std::vector<std::vector<float>> & logits,
              size_t                          & mem_per_token);

bark_vocab::id gpt_sample(
              std::vector<float>          & logits,
              std::mt19937                & rng,
              float temp,
              float * eos_p);

bool bark_model_load(const std::string & dirname, bark_model & model);

bool bark_vocab_load(const std::string & fname, bark_vocab& vocab, int32_t expected_size);

void bert_tokenize(
    const bark_vocab& vocab,
    const char * text,
    int32_t * tokens,
    int32_t * n_tokens,
    int32_t n_max_tokens);

bool bark_generate_audio(
        bark_model model,
        const bark_vocab& vocab,
        const char * text,
        const int n_threads);

bark_sequence bark_forward_text_encoder(
    const bark_sequence & tokens,
    const gpt_model model,
    std::mt19937 & rng,
    const int n_threads,
    const float temp,
    const float min_eos_p);

bark_codes bark_forward_coarse_encoder(
    const bark_sequence & tokens,
    const gpt_model model,
    std::mt19937 & rng,
    const int n_threads,
    const float temp,
    const int max_coarse_history,
    const int sliding_window_size);

bark_codes bark_forward_fine_encoder(
    const bark_codes & tokens,
    const gpt_model model,
    std::mt19937 & rng,
    const int n_threads,
    const float temp);

audio_arr_t bark_forward_encodec(
    const bark_codes & tokens,
    const encodec_model model,
    const int n_threads);

struct bark_progress {
    float current = 0.0f;
    const char * func = NULL;

    bark_progress() {}

    void callback(float progress) {
        float percentage = progress * 100;
        if (percentage == 0.0f && func != NULL) {
            fprintf(stderr, "%s: ", func);
        }
        while (percentage > current) {
            current = percentage;
            fprintf(stderr, ".");
            fflush(stderr);
            if (percentage >= 100) {
                fprintf(stderr, "\n");
            }
        }
    }
};
