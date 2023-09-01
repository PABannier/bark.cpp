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

    struct bark_context;
    struct bark_progress;

    struct bark_model;
    struct bark_vocab;

    struct bark_params {
        int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

        std::string model = "./ggml_weights/";  // weights location

        int32_t seed = 0;

        std::string prompt;  // user prompt

        std::string dest_wav_path = "./output.wav";
    };

    typedef int32_t bark_token;

    typedef std::vector<bark_token>              bark_sequence;
    typedef std::vector<std::vector<bark_token>> bark_codes;
    typedef std::vector<float>                   audio_arr_t;

    struct gpt_hparams;
    struct gpt_layer;
    struct gpt_model;

    BARK_API struct bark_context * bark_new_context_with_model(struct bark_model * model);

    BARK_API void bark_free(bark_context * ctx);

    BARK_API void bark_free_model(struct bark_model * ctx);

    BARK_API bool bark_generate_audio(
            struct bark_context * ctx,
                     const char * text,
                    std::string & dest_wav_path,
                            int   n_threads);

    BARK_API struct bark_model * bark_load_model_from_file(const std::string & dirname);

    BARK_API bool bark_model_quantize(
              const std::string & fname_inp,
              const std::string & fname_out,
                     ggml_ftype   ftype);

    BARK_API bool bark_vocab_load(const std::string & fname, bark_vocab& vocab, int32_t expected_size);

    BARK_API bool bark_params_parse(int argc, char ** argv, bark_params & params);

    BARK_API void bark_print_usage(char ** argv, const bark_params & params);

#ifdef __cplusplus
}
#endif

#ifdef BARK_API_INTERNAL

    // 
    // Internal API for tests
    // 

    bool gpt_model_load(const std::string& fname, gpt_model& model);

    bool gpt_eval(
             gpt_model & model,
        bark_token * tokens,
                   int   n_tokens,
                 float * logits,
                   int * n_past,
                  bool   merge_ctx,
                   int   n_threads);

    bool fine_gpt_eval(
             gpt_model & model,
        bark_token * tokens,
                   int   n_tokens,
                 float * logits,
                   int   n_threads,
                   int   codebook_ix);

    void bert_tokenize(
      const bark_vocab & vocab,
            const char * text,
               int32_t * tokens,
               int32_t * n_tokens,
               int32_t   n_max_tokens);

    void bark_forward_text_encoder(
        struct bark_context * ctx,
                      float   temp,
                      float   min_eos_p,
                        int   n_threads);

    void bark_forward_coarse_encoder(
        struct bark_context * ctx,
                        int   max_coarse_history,
                        int   sliding_window_size,
                      float   temp,
                        int   n_threads);

    void bark_forward_fine_encoder(
        struct bark_context * ctx,
                      float   temp, 
                        int   n_threads);

    void bark_forward_encodec(struct bark_context * ctx);

    void print_tensor(struct ggml_tensor * a);

    void read_tensor_from_file(std::ifstream & fin, struct ggml_tensor * t);

    bool allequal(struct ggml_tensor * a, struct ggml_tensor * b, std::string test_name);

    bool allclose(struct ggml_tensor * a, struct ggml_tensor * b, float tol, std::string test_name);

#endif // BARK_API_INTERNAL

#endif  // BARK_H