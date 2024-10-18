/*
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2024 Pierre-Antoine Bannier                                        │
│                                                                              │
│ Permission to use, copy, modify, and/or distribute this software for         │
│ any purpose with or without fee is hereby granted, provided that the         │
│ above copyright notice and this permission notice appear in all copies.      │
│                                                                              │
│ THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL                │
│ WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED                │
│ WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE             │
│ AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL         │
│ DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR        │
│ PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER               │
│ TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR             │
│ PERFORMANCE OF THIS SOFTWARE.                                                │
╚─────────────────────────────────────────────────────────────────────────────*/
#pragma once

#include "encodec.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef _WIN32
    #ifdef EXPORTING_BARK
        #define BARK_API __declspec(dllexport)
    #else
        #define BARK_API __declspec(dllimport)
    #endif
#else
    #define BARK_API
#endif

#ifdef __cplusplus
extern "C" {
#endif
    enum bark_verbosity_level {
        LOW    = 0,
        MEDIUM = 1,
        HIGH   = 2,
    };

    enum bark_encoding_step {
        SEMANTIC = 0,
        COARSE   = 1,
        FINE     = 2,
    };

    struct bark_context;
    struct bark_model;

    // Holds the vocabulary for the semantic encoder
    struct bark_vocab;

    // Define the GPT architecture for the 3 encoders
    struct gpt_model;

    typedef void (*bark_progress_callback)(struct bark_context * bctx, enum bark_encoding_step step, int progress, void * user_data);

    struct bark_statistics {
        // Time to load model weights
        int64_t t_load_us;
        // Time to generate audio
        int64_t t_eval_us;

        // Time to generate semantic tokens
        int64_t t_semantic_us;
        // Time to generate coarse tokens
        int64_t t_coarse_us;
        // Time to generate fine tokens
        int64_t t_fine_us;

        // Number of semantic tokens sampled
        int32_t n_sample_semantic;
        // Number of coarse tokens sampled
        int32_t n_sample_coarse;
        // Number of fine tokens sampled
        int32_t n_sample_fine;
    };

    struct bark_context_params {
        // Verbosity level
        enum bark_verbosity_level verbosity;

        // Temperature for sampling (text and coarse encoders)
        float temp;
        // Temperature for sampling (fine encoder)
        float fine_temp;

        // Minimum probability for EOS token (text encoder)
        float min_eos_p;
        // Sliding window size for coarse encoder
        int32_t sliding_window_size;
        // Max history for coarse encoder
        int32_t max_coarse_history;

        // Sample rate
        int32_t sample_rate;
        // Target bandwidth
        int32_t target_bandwidth;

        // CLS token ID
        int32_t cls_token_id;
        // SEP token ID
        int32_t sep_token_id;

        // Maximum number of semantic tokens to generate
        int32_t n_steps_text_encoder;

        // Text PAD token ID
        int32_t text_pad_token;
        // Text encoding offset
        int32_t text_encoding_offset;

        // Semantic frequency rate
        float semantic_rate_hz;
        // Semantic PAD token ID
        int32_t semantic_pad_token;
        // Vocabulary size in semantic encoder
        int32_t semantic_vocab_size;
        // Semantic infernce token ID
        int32_t semantic_infer_token;

        // Coarse frequency rate
        float coarse_rate_hz;
        // Coarse infer token ID
        int32_t coarse_infer_token;
        // Coarse semantic pad token ID
        int32_t coarse_semantic_pad_token;

        // Number of codebooks in coarse encoder
        int32_t n_coarse_codebooks;
        // Number of codebooks in fine encoder
        int32_t n_fine_codebooks;
        // Dimension of the codes
        int32_t codebook_size;

        // called on each progress update
        bark_progress_callback progress_callback;
        void * progress_callback_user_data;
    };

    /**
     * @brief Returns the default parameters for a bark context.
     *
     * @return bark_context_params The default parameters for a bark context.
     */
    BARK_API struct bark_context_params bark_context_default_params(void);

    /**
     * Loads a Bark model from the specified file path with the given parameters.
     *
     * @param model_path The directory path of the bark model to load.
     * @param params     The parameters to use for the Bark model.
     * @param seed       The seed to use for random number generation.
     * @return A pointer to the loaded bark model context.
     */
    BARK_API struct bark_context *bark_load_model(
        const char *model_path,
        struct bark_context_params params,
        uint32_t seed);

    /**
     * Generates an audio file from the given text using the specified Bark context.
     *
     * @param bctx The Bark context to use for generating the audio.
     * @param text The text to generate audio from.
     * @param n_threads The number of threads to use for generating the audio.
     * @return An integer indicating the success of the audio generation process.
     */
    BARK_API bool bark_generate_audio(
        struct bark_context *bctx,
        const char *text,
        int n_threads);

    /**
     * Retrieves the audio data generated by the Bark context.
     *
     * @param bctx The Bark context to use for generating the audio.
     * @return A pointer to the audio data generated by the Bark context.
     */
    BARK_API float *bark_get_audio_data(
        struct bark_context *bctx);

    /**
     * Retrieves the audio data generated by the Bark context.
     *
     * @param bctx The Bark context to use for generating the audio.
     * @return The size of the audio data generated by the Bark context.
     */
    BARK_API int bark_get_audio_data_size(
        struct bark_context *bctx);

    /**
     * Retrieves the load time of the last audio generation round.
     *
     * @param bctx The Bark context to use for generating the audio.
     * @return A struct containing the statistics of the last audio generation round.
     */
    BARK_API int64_t bark_get_load_time(
        struct bark_context *bctx);

    /**
     * Retrieves the evaluation time of the last audio generation round.
     *
     * @param bctx The Bark context to use for generating the audio.
     * @return A struct containing the statistics of the last audio generation round.
     */
    BARK_API int64_t bark_get_eval_time(
        struct bark_context *bctx);

    /**
     * Reset the statistics of the last audio generation round.
     *
     * @param bctx The Bark context to use for generating the audio.
     * @return A struct containing the statistics of the last audio generation round.
     */
    BARK_API void bark_reset_statistics(
        struct bark_context *bctx);

    /**
     * Quantizes a bark model and saves the result to a file.
     *
     * @param fname_inp The name of the input file containing the BARK model.
     * @param fname_out The name of the output file to save the quantized model to.
     * @param ftype The type of the model's floating-point values.
     * @return True if the model was successfully quantized and saved, false otherwise.
     */
    BARK_API bool bark_model_quantize(
        const char *fname_inp,
        const char *fname_out,
        enum ggml_ftype ftype);

    /**
     * @brief Frees the memory allocated for a bark context.
     *
     * @param bctx The bark context to free.
     */
    BARK_API void bark_free(
        struct bark_context *bctx);

#ifdef __cplusplus
}
#endif
