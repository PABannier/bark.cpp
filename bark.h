#ifndef BARK_H
#define BARK_H

#include "encodec.h"

#include <map>
#include <random>
#include <thread>
#include <vector>

#ifdef BARK_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef BARK_BUILD
#            define BARK_API __declspec(dllexport)
#        else
#            define BARK_API __declspec(dllimport)
#        endif
#    else
#        define BARK_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define BARK_API
#endif

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


#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //

    typedef int32_t bark_token;

    struct bark_context;
    struct bark_progress;

    struct bark_context_params {
        uint32_t seed; // RNG seed

        float temp;      // Temperature for sampling (text and coarse encoders)
        float fine_temp; // Temperature for sampling (fine encoder)

        float min_eos_p;         // Minimum probability for EOS token (text encoder)
        int sliding_window_size; // Sliding window size for coarse encoder
        int max_coarse_history;  // Max history for coarse encoder
    };

    struct bark_model;
    struct bark_vocab;

    struct gpt_hparams;
    struct gpt_layer;
    struct gpt_model;

    BARK_API struct bark_context * bark_new_context_with_model(
               struct bark_model * model,
      struct bark_context_params   params);

    BARK_API void bark_seed_rng(struct bark_context * ctx, int32_t seed);

    BARK_API struct bark_context_params bark_context_default_params();

    BARK_API void bark_free(struct bark_context * ctx);

    BARK_API void bark_free_model(struct bark_model * ctx);

    BARK_API int bark_generate_audio(
            struct bark_context * ctx,
                     const char * text,
                     const char * dest_wav_path,
                            int   n_threads);

    BARK_API struct bark_model * bark_load_model_from_file(const char * dirname);

    BARK_API int bark_model_quantize(
                     const char * fname_inp,
                     const char * fname_out,
                     ggml_ftype   ftype);

    BARK_API int bark_vocab_load(
                     const char * fname,
                     bark_vocab * vocab,
                        int32_t   expected_size);

#ifdef __cplusplus
}
#endif

#ifdef BARK_API_INTERNAL

    //
    // Internal API for tests
    //

    int encodec_eval(
            const std::vector<std::vector<int32_t>> & tokens,
                                      encodec_model & model,
                                 std::vector<float> & audio_arr);

    int gpt_model_load(const std::string& fname, gpt_model& model);

    int gpt_eval(
                  gpt_model * model,
                 bark_token * tokens,
                        int   n_tokens,
                      float * logits,
                        int * n_past,
                       bool   merge_ctx,
                        int   n_threads);

    bool fine_gpt_eval(
                  gpt_model * model,
                 bark_token * tokens,
                        int   n_tokens,
                      float * logits,
                        int   n_threads,
                        int   codebook_ix);

    void bert_tokenize(
           const bark_vocab * vocab,
                 const char * text,
                    int32_t * tokens,
                    int32_t * n_tokens,
                    int32_t   n_max_tokens);

    void bark_forward_text_encoder(
        struct bark_context * ctx,
                        int   n_threads);

    void bark_forward_coarse_encoder(
        struct bark_context * ctx,
                        int   n_threads);

    void bark_forward_fine_encoder(
        struct bark_context * ctx,
                        int   n_threads);

    void bark_forward_encodec(struct bark_context * ctx);

#endif // BARK_API_INTERNAL

#endif  // BARK_H