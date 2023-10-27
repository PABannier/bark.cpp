#include "ggml.h"

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

struct bark_context;
struct bark_progress;

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

struct bark_model;
struct bark_vocab;

struct gpt_hparams;
struct gpt_layer;
struct gpt_model;

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
           const std::string & model_path,
   const bark_context_params & params);

/**
 * Generates an audio file from the given text using the specified Bark context.
 * 
 * @param bctx The Bark context to use for generating the audio.
 * @param text The text to generate audio from.
 * @param dest_wav_path The path to save the generated audio file.
 * @param n_threads The number of threads to use for generating the audio.
 * @return An integer indicating the success of the audio generation process.
 */
BARK_API int bark_generate_audio(
         struct bark_context * bctx,
           const std::string & text,
           const std::string & dest_wav_path,
                         int   n_threads);

/**
 * @brief Frees the memory allocated for a bark context.
 * 
 * @param bctx The bark context to free.
 */
BARK_API void bark_free(struct bark_context * bctx);
