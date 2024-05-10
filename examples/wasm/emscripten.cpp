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
                                     bark_verbosity_level verbosity = bark_verbosity_level::LOW;
                                     bark_context_params ctx_params = bark_context_default_params();
                                     ctx_params.verbosity = verbosity;
                                     g_contexts[i] = bark_load_model(path_model.c_str(), ctx_params, 0 /* seed */);
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

    emscripten::function("generate", emscripten::optional_override([](size_t index, const std::string &prompt, int n_threads) {
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
                                        n_threads, std::thread::hardware_concurrency());

                                 printf("\n");
                             }

                             // run the worker and generate audio
                             {
                                 g_worker = std::thread([index, prompt, n_threads]() {
                                    ggml_time_init();
                                    const int64_t t_main_start_us = ggml_time_us();

                                    if (!bark_generate_audio(g_contexts[index], prompt.c_str(), n_threads)) {
                                        printf("An error occured.");
                                    }

                                    const int64_t t_main_end_us = ggml_time_us();
                                    const int64_t t_load_us = bark_get_load_time(g_contexts[index]);
                                    const int64_t t_eval_us = bark_get_eval_time(g_contexts[index]);

                                    printf("\n\n");
                                    printf("%s:     load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
                                    printf("%s:     eval time = %8.2f ms\n", __func__, t_eval_us / 1000.0f);
                                    printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
                                 });
                             }

                             return 0;
                         }));

    emscripten::function("getAudioBuffer", emscripten::optional_override([](size_t index) -> emscripten::val {
                             if (index >= g_contexts.size() || g_contexts[index] == nullptr) {
                                 return emscripten::val::null();
                             }

                             const float * audio_data = bark_get_audio_data(g_contexts[index]);
                             if (audio_data == nullptr) {
                                printf("Could not retrieve audio data");
                                return emscripten::val::null();
                             }

                             const int n_samples = bark_get_audio_data_size(g_contexts[index]);

                             emscripten::val result = emscripten::val::object();
                             result.set("ptr", reinterpret_cast<uintptr_t>(audio_data));
                             result.set("size", n_samples * sizeof(float));

                             return result;
                         }));
}