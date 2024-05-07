#include <stdint.h>

#if __cplusplus
extern "C" {
#endif

struct bark_coreml_context;

struct bark_coreml_context* bark_coreml_init(const char * model_path);
void bark_coreml_free(struct bark_coreml_context* ctx);

void bark_coreml_generate_audio(
        const bark_coreml_context * ctx,
                       const char * prompt,
                            float * out_audio);

#if __cplusplus
}
#endif
