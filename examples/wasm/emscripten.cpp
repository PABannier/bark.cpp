#include <emscripten.h>
#include <emscripten/bind.h>

#include <thread>
#include <vector>

#include "bark.h"
#include "ggml.h"

std::thread g_worker;
std::vector<struct bark_context *> g_contexts(4, nullptr);

EMSCRIPTEN_BINDINGS(bark) {
    emscripten::function("init", emscripten::optional_override([](const std::string &path_model) {
                             if (g_worker.joinable()) {
                                 g_worker.join();
                             }

                             for (size_t i = 0; i < g_contexts.size(); i++) {
                                 if (g_contexts[i] == nullptr) {
                                     g_contexts[i] = bark_load_model(path_model.c_str(), bark_verbosity_level::LOW);
                                     if (g_contexts[i] != nullptr) {
                                         return i + 1;
                                     } else {
                                         return (size_t)0;
                                     }
                                 }
                             }

                             return (size_t)0;
                         }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
                             if (g_worker.joinable()) {
                                 g_worker.join();
                             }

                             --index;

                             if (index < g_contexts.size()) {
                                 bark_free(g_contexts[index]);
                                 g_contexts[index] = nullptr;
                             }
                         }));

    emscripten::function("generate", emscripten::optional_override([](size_t index, const std::string &prompt, const std::string &path_wav, int nthreads) {
                             if (g_worker.joinable()) {
                                 g_worker.join();
                             }

                             --index;

                             if (index >= g_contexts.size()) {
                                 return -1;
                             }

                             if (g_contexts[index] == nullptr) {
                                 return -2;
                             }

                             // print system information
                             {
                                 printf("system_info: n_threads = %d / %d \n",
                                        params.n_threads, std::thread::hardware_concurrency());

                                 printf("\n");
                             }

                             // run the worker and generate audio
                             {
                                 g_worker = std::thread([index, prompt, n_threads]() {
                                     ggml_time_init();
                                     const int64_t t_main_start_us = ggml_time_us();

                                     if (!bark_generate_audio(g_contexts[index], prompt, dest_wav_path, n_threads)) {
                                         printf("An error occured.");
                                     }

                                     const int64_t t_main_end_us = ggml_time_us();

                                     printf("\n\n");
                                     printf("%s:     load time = %8.2f ms\n", __func__, bctx->t_load_us / 1000.0f);
                                     printf("%s:     eval time = %8.2f ms\n", __func__, bctx->t_eval_us / 1000.0f);
                                     printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
                                 });
                             }
                         }));
}