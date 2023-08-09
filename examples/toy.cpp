#include <cstdio>
#include <vector>

#include "ggml.h"
#include "encodec.h"

void display_tensor_content(struct ggml_tensor * tensor) {
    if (tensor) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int i = 0; i < tensor->ne[0]; i++) {
                float * v = (float *) ((char *)tensor->data + j*tensor->nb[1] + i*tensor->nb[0]);
                fprintf(stderr, "%.4f ", *v);
            }
            fprintf(stderr, "\n");
        }
    }
    fprintf(stderr, "\n");
}

int main() {
    static size_t n_threads = 4;
    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    static int N = 10;
    static int n_channels = 4;

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, n_channels);
    for (int i = 0; i < n_channels; i++) {
        std::vector<float> _tmp(N);
        for (int j = 0; j < (int) _tmp.size(); j++) { _tmp[i] = (i+1)*(j+1)/10.f; }
        int offset = i*N*sizeof(float);
        memcpy((void *) ((char *)input->data + offset), _tmp.data(), N*sizeof(float));
    }

    // define operations
    int padding_left  = 3;
    int padding_right = 2;
    struct ggml_tensor * out = ggml_pad_reflec_1d(ctx0, input, padding_left, padding_right);

    // run computations
    ggml_build_forward_expand(&gf, out);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    // display
    fprintf(stderr, "input=\n");
    display_tensor_content(input);
    fprintf(stderr, "output=\n");
    display_tensor_content(out);

    ggml_free(ctx0);

    return 0;
}


// int main() {
//     static size_t n_threads = 4;
//     static size_t buf_size = 256u*1024*1024;
//     static void * buf = malloc(buf_size);

//     struct ggml_init_params params = {
//         /*.mem_size   =*/ buf_size,
//         /*.mem_buffer =*/ buf,
//         /*.no_alloc   =*/ false,
//     };

//     struct ggml_context * ctx0 = ggml_init(params);
//     struct ggml_cgraph gf = {};

//     static int N = 10;
//     static int n_channels = 4;

//     struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, n_channels);
//     for (int i = 0; i < n_channels; i++) {
//         std::vector<float> _tmp(N, i+1);
//         int offset = i*N*sizeof(float);
//         memcpy((void *) ((char *)input->data + offset), _tmp.data(), N*sizeof(float));
//     }

//     // constant padding
//     int extra_pad = 3;
//     struct ggml_tensor * out = ggml_new_tensor_2d(ctx0, input->type, N + extra_pad, n_channels);
//     ggml_set_zero(out);
//     out = ggml_set_2d(ctx0, out, input, out->nb[1], 0);

//     // run computations
//     ggml_build_forward_expand(&gf, out);
//     ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

//     // display
//     fprintf(stderr, "input=\n");
//     display_tensor_content(input);
//     fprintf(stderr, "output=\n");
//     display_tensor_content(out);

//     ggml_free(ctx0);

//     return 0;
// }
