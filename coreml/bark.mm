#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "bark.h"
#import "bark-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

struct bark_coreml_context {
    const void * data;
};

struct bark_coreml_context * bark_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // config.computeUnits = MLComputeUnitsCPUAndGPU;
    //config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.computeUnits = MLComputeUnitsAll;

    const void * data = CFBridgingRetain([[bark_impl alloc] initWithContentsOfURL:url_model configuration:config error:nil]);

    if (data == NULL) {
        return NULL;
    }

    bark_coreml_context * ctx = new bark_coreml_context;

    ctx->data = data;

    return ctx;
}

void bark_coreml_free(struct bark_coreml_free * ctx) {
    CFRelease(ctx->data);
    delete ctx;
}

void bark_coreml_generate_audio(
           const bark_coreml_context * ctx,
                          const char * prompt,
                               float * out_audio) {
    NSString * prompt_str = [[NSString alloc] initWithUTF8String:prompt]; 

    @autoreleasepool {
       // whisper_encoder_implOutput * outCoreML = [(__bridge id) ctx->data predictionFromLogmel_data:inMultiArray error:nil];
       bark_implOutput * outCoreML = [(__bridge id) ctx->data generateAudioFromPrompt_data:prompt_str error:nil];

        memcpy(out_audio, outCoreML.output.dataPointer, outCoreML.output.count * sizeof(float));
    }
}

#if __cplusplus
}
#endif
