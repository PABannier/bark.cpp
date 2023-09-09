#include "ggml.h"

#include <random>
#include <vector>

float rand_float(float minv, float maxv) {
    return ((float(rand()) / float(RAND_MAX)) * (maxv - minv)) + minv;
}

void dump_tensor(struct ggml_tensor * a, bool print_val) {
    float sum = 0;
    for (int i = 0; i < a->ne[3]; i++) {
        for (int j = 0; j < a->ne[2]; j++) {
            for (int k = 0; k < a->ne[1]; k++) {
                for (int l = 0; l < a->ne[0]; l++) {
                    if (a->type == GGML_TYPE_F32) {
                        float * aval = (float *) (
                            (char *) a->data + i*a->nb[3] + j*a->nb[2] + k*a->nb[1] + l*a->nb[0]);
                        sum += *aval;
                        if (print_val)
                            printf("%.4f ", *aval);
                    } else if (a->type == GGML_TYPE_I32) {
                        int32_t * aval = (int32_t *) (
                            (char *) a->data + i*a->nb[3] + j*a->nb[2] + k*a->nb[1] + l*a->nb[0]);
                        sum += *aval;
                        if (print_val)
                            printf("%d ", *aval);
                    } else {
                        throw;
                    }
                }
                if (print_val)
                    printf("\n");
            }
            if (print_val)
                printf("\n\n");
        }
    }
    printf("sum=%.4f\n", sum);
    printf("ne=[%lld, %lld, %lld, %lld]\n", a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
    printf("nb=[%zu, %zu, %zu, %zu]\n", a->nb[0], a->nb[1], a->nb[2], a->nb[3]);
    printf("type=%u\n", a->type);
}

void populate(struct ggml_tensor * t) {
    for (int k = 0; k < t->ne[2]; k++) {
        for (int j = 0; j < t->ne[1]; j++) {
            for (int i = 0; i < t->ne[0]; i++) {
                float r = rand_float(-0.5f, 0.5f);
                switch (t->type) {
                    case GGML_TYPE_F32:
                    {
                        float * v = (float *)((char *) t->data + k*t->nb[2] + j*t->nb[1] + i*t->nb[0]);
                        *v = r;
                    } break;
                    case GGML_TYPE_F16:
                    {
                        ggml_fp16_t * v = (ggml_fp16_t *)((char *) t->data + k*t->nb[2] + j*t->nb[1] + i*t->nb[0]);
                        *v = ggml_fp32_to_fp16(r);
                    } break;
                    default: {
                        assert(false);
                    } break;
                }
            }
        }
    }
}

void run_conv_test(int seq_length, int kernel_size, int in_channels, int out_channels, int stride, bool fp16) {
    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    ggml_type type = fp16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_length, in_channels);
    populate(inp);

    struct ggml_tensor * conv_w = ggml_new_tensor_3d(ctx0, type, kernel_size, in_channels, out_channels);
    populate(conv_w);

    struct ggml_tensor * out = ggml_conv_1d(ctx0, conv_w, inp, stride, 0, 1);

    ggml_build_forward_expand(&gf, out);
    ggml_graph_compute_with_ctx(ctx0, &gf, 1);

    printf("\n");
    printf("%s\n", __func__);
    dump_tensor(out, false  /* print_val */);
}

void run_conv_transpose_test(int seq_length, int kernel_size, int in_channels, int out_channels, int stride, bool fp16) {
    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    ggml_type type = fp16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_length, in_channels);
    populate(inp);

    struct ggml_tensor * conv_w = ggml_new_tensor_3d(ctx0, type, kernel_size, out_channels, in_channels);
    populate(conv_w);

    struct ggml_tensor * out = ggml_conv_transpose_1d(ctx0, conv_w, inp, stride, 0 /* p0 */, 1 /* d0 */);

    ggml_build_forward_expand(&gf, out);
    ggml_graph_compute_with_ctx(ctx0, &gf, 1);

    printf("\n");
    printf("%s\n", __func__);
    dump_tensor(out, false  /* print_val */);
}

int main() {
    srand(0);

    // conv

    // small configuration
    {
        // fp16
        run_conv_test(
            10 /* seq_length */, 
            3 /* kernel_size */, 
            2 /* in_channels */, 
            4 /* out_channels */, 
            1 /* stride */,
            true /* fp16 */);

        // fp32
        run_conv_test(
            10 /* seq_length */, 
            3 /* kernel_size */, 
            2 /* in_channels */, 
            4 /* out_channels */, 
            1 /* stride */,
            false /* fp16 */);
    }

    // large configuration
    {
        // fp16
        run_conv_test(
            1024 /* seq_length */, 
            16 /* kernel_size */, 
            14 /* in_channels */, 
            4 /* out_channels */, 
            8 /* stride */,
            true /* fp16 */);

        // fp32
        run_conv_test(
            1024 /* seq_length */, 
            16 /* kernel_size */, 
            14 /* in_channels */, 
            4 /* out_channels */, 
            8 /* stride */,
            false /* fp16 */);
    }

    // conv transpose

    // small configuration
    {
        // fp16
        run_conv_transpose_test(
            10 /* seq_length */, 
            3 /* kernel_size */, 
            2 /* in_channels */, 
            4 /* out_channels */, 
            1 /* stride */,
            true /* fp16 */);

        // fp32
        run_conv_transpose_test(
            10 /* seq_length */, 
            3 /* kernel_size */, 
            2 /* in_channels */, 
            4 /* out_channels */, 
            1 /* stride */,
            false /* fp16 */);
    }

    // large configuration
    {
        // fp16
        run_conv_transpose_test(
            1024 /* seq_length */, 
            16 /* kernel_size */, 
            14 /* in_channels */, 
            4 /* out_channels */, 
            8 /* stride */,
            true /* fp16 */);

        // fp32
        run_conv_transpose_test(
            1024 /* seq_length */, 
            16 /* kernel_size */, 
            14 /* in_channels */, 
            4 /* out_channels */, 
            8 /* stride */,
            false /* fp16 */);
    }

    
    return 0;
}