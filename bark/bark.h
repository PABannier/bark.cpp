#include "ggml.h"
#include "ggml-backend.h"

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

typedef int32_t bark_token;

typedef std::vector<int32_t> bark_sequence;
typedef std::vector<std::vector<int32_t>> bark_codes;

struct gpt_hparams {
    int32_t n_in_vocab;
    int32_t n_out_vocab;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_embd;
    int32_t block_size;
    int32_t n_lm_heads;
    int32_t n_wtes;
    int32_t ftype;
    int32_t bias;

    int32_t n_codes_given = 1;
};

struct bark_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

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

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wpe;                       //  position embedding
    std::vector<struct ggml_tensor *> wtes;         //     token embedding
    std::vector<struct ggml_tensor *> lm_heads;     // language model head

    std::vector<gpt_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;
    ggml_backend_buffer_t buffer_kv;

    std::map<std::string, struct ggml_tensor *> tensors;

    //
    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;
    int64_t t_main_us    = 0;

    //
    int64_t n_sample  = 0;
    int64_t n_predict = 0;

    //
    int64_t memsize = 0;
    size_t mem_per_token = 0;
};

struct bark_model {
    // encoder
    gpt_model coarse_model;
    gpt_model   fine_model;
    gpt_model   text_model;

    // vocab
    bark_vocab vocab;
};

struct bark_context_params {
    uint32_t seed; // RNG seed

    // Temperature for sampling (text and coarse encoders)
    float temp;
    // Temperature for sampling (fine encoder)
    float fine_temp;

    // Minimum probability for EOS token (text encoder)
    float min_eos_p;
    // Sliding window size for coarse encoder
    int sliding_window_size;
    // Max history for coarse encoder
    int max_coarse_history;
};

struct bark_context {
    bark_model model;

    // buffer for model evaluation
    ggml_backend_buffer_t buf_compute;

    // custom allocator
    struct ggml_allocr * allocr = NULL;

    std::mt19937 rng;

    bark_sequence tokens;
    bark_sequence semantic_tokens;

    bark_codes coarse_tokens;
    bark_codes fine_tokens;

    std::vector<float> audio_arr;

    // hyperparameters
    bark_context_params params;

    // statistics
    int64_t t_load_us = 0;
    int64_t t_eval_us = 0;

};

/**
 * @brief Returns the default parameters for a bark context.
 *
 * @return bark_context_params The default parameters for a bark context.
 */
BARK_API struct bark_context_params bark_context_default_params(void);

/**
 * Loads a BARK model from the specified file path with the given parameters.
 *
 * @param model_path The directory path of the bark model to load.
 * @param params The parameters to use when loading the bark model.
 * @return A pointer to the loaded bark model context.
 */
BARK_API struct bark_context * bark_load_model(
           const std::string & model_path);

/**
 * Generates an audio file from the given text using the specified Bark context.
 *
 * @param bctx The Bark context to use for generating the audio.
 * @param text The text to generate audio from.
 * @param dest_wav_path The path to save the generated audio file.
 * @param n_threads The number of threads to use for generating the audio.
 * @return An integer indicating the success of the audio generation process.
 */
BARK_API bool bark_generate_audio(
                bark_context * bctx,
                 std::string & text,
                 std::string & dest_wav_path,
                         int   n_threads);

/**
 * Quantizes a bark model and saves the result to a file.
 *
 * @param fname_inp The name of the input file containing the BARK model.
 * @param fname_out The name of the output file to save the quantized model to.
 * @param ftype The type of the model's floating-point values.
 * @return True if the model was successfully quantized and saved, false otherwise.
 */
BARK_API bool bark_model_quantize(
           const std::string & fname_inp,
           const std::string & fname_out,
                  ggml_ftype   ftype);

/**
 * @brief Frees the memory allocated for a bark context.
 *
 * @param bctx The bark context to free.
 */
BARK_API void bark_free(
        struct bark_context * bctx);

/**
 * Loads a vocabulary from a file.
 *
 * @param fname The name of the file to load the vocabulary from.
 * @param vocab A pointer to the bark_vocab struct to store the loaded vocabulary in.
 * @param expected_size The expected size of the vocabulary.
 * @return true if the vocabulary was loaded successfully, false otherwise.
 */
bool bark_vocab_load(
     const std::string & fname,
            bark_vocab * vocab,
               int32_t   expected_size);

/**
 * Tokenizes the input text using the provided vocabulary.
 *
 * @param vocab Pointer to the vocabulary to use for tokenization.
 * @param text The input text to tokenize.
 * @param tokens Pointer to an array where the resulting tokens will be stored.
 * @param n_tokens Pointer to an integer where the number of resulting tokens will be stored.
 * @param n_max_tokens The maximum number of tokens that can be stored in the tokens array.
 */
void bert_tokenize(
        const bark_vocab * vocab,
              const char * text,
                 int32_t * tokens,
                 int32_t * n_tokens,
                 int32_t   n_max_tokens);


/**
 * Encodes the input text using the forward algorithm.
 * 
 * @param bctx A pointer to the bark context struct.
 * @param n_threads The number of threads to use for encoding.
 * @return Returns true if the encoding was successful, false otherwise.
 */
bool bark_forward_text_encoder(
     struct bark_context * bctx, 
                     int   n_threads);