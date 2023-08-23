#include "encodec.h"
#include "ggml.h"
#include "bark-util.h"

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// Implementation of the 1d transposed convolution by applying a 1d convolution on a
// reshaped input.
// Source: https://arxiv.org/pdf/1603.07285
// Source: https://leimao.github.io/blog/Transposed-Convolution-As-Convolution/
static struct ggml_tensor * transpose_conv_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * inp,
        struct ggml_tensor  * ker,
                       int    stride) {
    int seq_length  = inp->ne[0];
    int kernel_size = ker->ne[0];

    BARK_ASSERT(kernel_size >= 1);
    BARK_ASSERT(stride > 0);

    // compute reshape statistics
    int s_prime = 1;
    int p_prime = kernel_size - 1;

    int out_seq_length = (seq_length - 1) * stride + (kernel_size - 1) + 1;

    int a = (out_seq_length - kernel_size) % stride;

    BARK_ASSERT(s_prime >= 0);
    BARK_ASSERT(p_prime >= 0);
    BARK_ASSERT(out_seq_length > 0);
    BARK_ASSERT(a >= 0);

    // the input and output channel need to be permuted
    // the spatial dimension need to be flipped
    ker = ggml_cont(ctx, ggml_permute(ctx, ker, 0, 2, 1, 3));
    ker = ggml_flip(ctx, ker);

    // interleaving input with "stride" 0s along the spatial dimension
    inp = ggml_interleave(ctx, inp, stride, 0.f);

    // padding the spatial dimension with "a" 0s
    struct ggml_tensor * padded_inp = ggml_new_tensor_2d(ctx, inp->type, inp->ne[0]+a, inp->ne[1]);
    padded_inp = ggml_set_zero(padded_inp);
    padded_inp = ggml_set_2d(ctx, padded_inp, inp, padded_inp->nb[1], 0);

    // perform convolution
    printf("s_prime=%d / p_prime=%d \n", s_prime, p_prime);
    printf("ker / ne = [%lld, %lld, %lld, %lld] / nb = [%zu, %zu, %zu, %zu]\n", ker->ne[0], ker->ne[1], ker->ne[2], ker->ne[3], ker->nb[0], ker->nb[1], ker->nb[2], ker->nb[3]);
    printf("inp / ne = [%lld, %lld, %lld, %lld] / nb = [%zu, %zu, %zu, %zu]\n", padded_inp->ne[0], padded_inp->ne[1], padded_inp->ne[2], padded_inp->ne[3], padded_inp->nb[0], padded_inp->nb[1], padded_inp->nb[2], padded_inp->nb[3]);
    struct ggml_tensor * out = ggml_conv_1d(ctx, ker, padded_inp, s_prime, p_prime, 1);

    return out;
}

static int get_extra_padding_for_conv_1d(ggml_tensor * inp, float kernel_size, float stride, float padding_total) {
    float length = inp->ne[0];
    float n_frames = (length - kernel_size + padding_total) / stride + 1.0f;
    int ideal_length = (ceilf(n_frames) - 1) * stride + (kernel_size - padding_total);
    return ideal_length - length;
}

static struct ggml_tensor * pad_1d(ggml_context * ctx0, ggml_tensor * inp, int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim = inp->ne[1];

    const int max_pad = std::max(padding_left, padding_right);
    int extra_pad = 0;

    if (length <= max_pad) {
        extra_pad = max_pad - length + 1;

        // constant padding
        struct ggml_tensor * out = ggml_new_tensor_2d(ctx0, inp->type, length+extra_pad, dim);
        out = ggml_set_zero(out);
        out = ggml_set_2d(ctx0, out, inp, out->nb[1], 0);
    }

    struct ggml_tensor * padded = ggml_pad_reflec_1d(ctx0, inp, padding_left, padding_right);

    const int end = padded->ne[0] - extra_pad;
    struct ggml_tensor *dest = ggml_view_2d(ctx0, padded, end, dim, padded->nb[1], 0);

    return dest;
}

static struct ggml_tensor * unpad_1d(ggml_context * ctx0, ggml_tensor * inp, int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim    = inp->ne[1];

    ENCODEC_ASSERT(padding_left  >= 0);
    ENCODEC_ASSERT(padding_right >= 0);
    ENCODEC_ASSERT(padding_left + padding_right <= length);

    int end = length - padding_right;

    int offset = padding_left * inp->nb[1];
    struct ggml_tensor * dst = ggml_view_2d(ctx0, inp, end, dim, inp->nb[1], offset);

    return dst;
}

static struct ggml_tensor * forward_pass_lstm_unilayer(
            struct ggml_context * ctx0,
            struct ggml_tensor * inp,
            struct ggml_tensor * weight_ih,
            struct ggml_tensor * weight_hh,
            struct ggml_tensor * bias_ih,
            struct ggml_tensor * bias_hh) {

    const int input_dim  = inp->ne[1];
    const int hidden_dim = weight_ih->ne[1]/4;
    const int seq_length = inp->ne[0];

    struct ggml_tensor * hs = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);

    struct ggml_tensor * c_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);
    struct ggml_tensor * h_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);

    h_t = ggml_set_zero(h_t);
    c_t = ggml_set_zero(c_t);

    struct ggml_tensor * current = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    for (int t = 0; t < seq_length; t++) {
        struct ggml_tensor * x_t = ggml_view_1d(ctx0, current, input_dim, t*current->nb[1]);

        struct ggml_tensor * inp_gates = ggml_mul_mat(ctx0, weight_ih, x_t);
        inp_gates = ggml_add(ctx0, inp_gates, bias_ih);

        struct ggml_tensor * hid_gates = ggml_mul_mat(ctx0, weight_hh, h_t);
        hid_gates = ggml_add(ctx0, hid_gates, bias_hh);

        struct ggml_tensor * out_gates = ggml_add(ctx0, inp_gates, hid_gates);

        struct ggml_tensor * i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 0*sizeof(float)*hidden_dim));
        struct ggml_tensor * f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 1*sizeof(float)*hidden_dim));
        struct ggml_tensor * g_t = ggml_tanh   (ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 2*sizeof(float)*hidden_dim));
        struct ggml_tensor * o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 3*sizeof(float)*hidden_dim));

        c_t = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_t), ggml_mul(ctx0, i_t, g_t));
        h_t = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_t));

        hs = ggml_set_1d(ctx0, hs, h_t, t*hs->nb[1]);
    }

    hs = ggml_cont(ctx0, ggml_transpose(ctx0, hs));

    return hs;
}

static struct ggml_tensor * strided_conv_1d(
            ggml_context * ctx0,
             ggml_tensor * inp,
             ggml_tensor * conv_w,
             ggml_tensor * conv_b,
                     int   stride) {
    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;
    int extra_padding = get_extra_padding_for_conv_1d(inp, kernel_size, stride, padding_total);

    struct ggml_tensor * padded_inp = pad_1d(ctx0, inp, padding_total, extra_padding);
    struct ggml_tensor * dst = ggml_conv_1d(ctx0, conv_w, padded_inp, stride, 0, 1);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    return dst;
}

static struct ggml_tensor * strided_conv_transpose_1d(
                ggml_context * ctx0,
                ggml_tensor * inp,
                ggml_tensor * conv_w,
                ggml_tensor * conv_b,
                        int   stride) {
    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;

    struct ggml_tensor * dst = transpose_conv_1d(ctx0, inp, conv_w, stride);
    return ggml_flip(ctx0, dst);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    int padding_right = ceilf(padding_total);
    int padding_left = padding_total - padding_right;

    struct ggml_tensor * unpadded = unpad_1d(ctx0, dst, padding_left, padding_right);
    unpadded = ggml_cont(ctx0, unpadded);

    return unpadded;
}

bool encodec_model_load(const std::string& fname, encodec_model& model) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic (i.e. ggml signature in hex format)
    {
        uint32_t magic;
        read_safe(fin, magic);
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    auto & ctx = model.ctx;
    size_t ctx_size = 0;

    // Evaluating context size
    {
        const auto & hparams = model.hparams;

        const int in_channels   = hparams.in_channels;
        const int hidden_dim    = hparams.hidden_dim;
        const int n_filters     = hparams.n_filters;
        const int kernel_size   = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int n_q           = hparams.n_q;
        const int n_bins        = hparams.n_bins;
        const int *ratios       = hparams.ratios;

        // decoder
        {
            // initial conv1d layer
            ctx_size += in_channels*n_filters*kernel_size*ggml_type_size(GGML_TYPE_F32);  // weight
            ctx_size +=                         n_filters*ggml_type_size(GGML_TYPE_F32);  //bias

            int mult = 1;  // scaling factor for hidden size

            for (int i = 0; i < 4; i++) {
                // conv1
                ctx_size += res_kernel_sz*(mult*n_filters)*(mult*n_filters/2)*ggml_type_size(GGML_TYPE_F32);  // weight
                ctx_size +=                                (mult*n_filters/2)*ggml_type_size(GGML_TYPE_F32);  // bias

                // conv2
                ctx_size += (mult*n_filters/2)*(mult*n_filters)*ggml_type_size(GGML_TYPE_F32);
                ctx_size +=                    (mult*n_filters)*ggml_type_size(GGML_TYPE_F32);

                // shortcut conv
                ctx_size += (mult*n_filters)*(mult*n_filters)*ggml_type_size(GGML_TYPE_F32);
                ctx_size +=                  (mult*n_filters)*ggml_type_size(GGML_TYPE_F32);

                // downsampling blocks
                ctx_size += (2*ratios[i])*(mult*n_filters)*(mult*n_filters*2)*ggml_type_size(GGML_TYPE_F32);
                ctx_size +=                                (mult*n_filters*2)*ggml_type_size(GGML_TYPE_F32);

                mult *= 2;
            }

            // lstm
            {
                // l0_ih, l0_hh, l1_ih, l1_hh all have the same shapes, hence 4
                ctx_size += 4*(mult*n_filters)*(4*mult*n_filters)*ggml_type_size(GGML_TYPE_F32);  // weight
                ctx_size +=                  4*(4*mult*n_filters)*ggml_type_size(GGML_TYPE_F32);  // bias
            }

            // final conv
            ctx_size += kernel_size*(mult*n_filters)*hidden_dim*ggml_type_size(GGML_TYPE_F32);
            ctx_size +=                              hidden_dim*ggml_type_size(GGML_TYPE_F32);
        }

        // quantizer
        {
            ctx_size += n_q*hidden_dim*n_bins; // embed
        }

        ctx_size += 10ull*MB;  // object overhead
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /* .mem_size   = */   ctx_size,
            /* .mem_buffer = */   NULL,
            /* .no_alloc   = */   false,
        };

        model.ctx = ggml_init(params);
        if(!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int in_channels   = hparams.in_channels;
        const int hidden_dim    = hparams.hidden_dim;
        const int n_filters     = hparams.n_filters;
        const int kernel_size   = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int n_q           = hparams.n_q;
        const int *ratios       = hparams.ratios;
        const int n_bins        = hparams.n_bins;

        // decoder
        {
            model.decoder.blocks.resize(4);

            int mult = 16;  // 2**len(ratios)

            model.decoder.init_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, hidden_dim, mult*n_filters);
            model.decoder.init_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

            model.tensors["decoder.model.0.conv.conv.weight"] = model.decoder.init_conv_w;
            model.tensors["decoder.model.0.conv.conv.bias"]   = model.decoder.init_conv_b;

            // LSTM
            model.decoder.lstm.l0_ih_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);
            model.decoder.lstm.l1_ih_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.weight_ih_l0"] = model.decoder.lstm.l0_ih_w;
            model.tensors["decoder.model.1.lstm.weight_ih_l1"] = model.decoder.lstm.l1_ih_w;

            model.decoder.lstm.l0_hh_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);
            model.decoder.lstm.l1_hh_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.weight_hh_l0"] = model.decoder.lstm.l0_hh_w;
            model.tensors["decoder.model.1.lstm.weight_hh_l1"] = model.decoder.lstm.l1_hh_w;

            model.decoder.lstm.l0_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.decoder.lstm.l1_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.bias_ih_l0"] = model.decoder.lstm.l0_ih_b;
            model.tensors["decoder.model.1.lstm.bias_ih_l1"] = model.decoder.lstm.l1_ih_b;

            model.decoder.lstm.l0_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.decoder.lstm.l1_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.bias_hh_l0"] = model.decoder.lstm.l0_hh_b;
            model.tensors["decoder.model.1.lstm.bias_hh_l1"] = model.decoder.lstm.l1_hh_b;

            for (int i = 0; i < 4; i++) {
                // upsampling
                model.decoder.blocks[i].us_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ratios[i]*2, mult*n_filters/2, mult*n_filters);
                model.decoder.blocks[i].us_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)) + ".convtr.convtr.weight"] = model.decoder.blocks[i].us_conv_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)) + ".convtr.convtr.bias"]   = model.decoder.blocks[i].us_conv_b;

                // conv1
                model.decoder.blocks[i].conv_1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, res_kernel_sz, mult*n_filters/2, mult*n_filters/4);
                model.decoder.blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/4);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.1.conv.conv.weight"] = model.decoder.blocks[i].conv_1_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.1.conv.conv.bias"]     = model.decoder.blocks[i].conv_1_b;

                // conv2
                model.decoder.blocks[i].conv_2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, mult*n_filters/4, mult*n_filters/2);
                model.decoder.blocks[i].conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.3.conv.conv.weight"] = model.decoder.blocks[i].conv_2_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.3.conv.conv.bias"]   = model.decoder.blocks[i].conv_2_b;

                // shortcut
                model.decoder.blocks[i].conv_sc_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, mult*n_filters/2, mult*n_filters/2);
                model.decoder.blocks[i].conv_sc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".shortcut.conv.conv.weight"] = model.decoder.blocks[i].conv_sc_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".shortcut.conv.conv.bias"]   = model.decoder.blocks[i].conv_sc_b;

                mult /= 2;
            }

            model.decoder.final_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, n_filters, in_channels);
            model.decoder.final_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            model.tensors["decoder.model.15.conv.conv.weight"] = model.decoder.final_conv_w;
            model.tensors["decoder.model.15.conv.conv.bias"]   = model.decoder.final_conv_b;
        }

        // quantizer
        {
            model.quantizer.blocks.resize(n_q);
            for (int i = 0; i < n_q; i++) {
                model.quantizer.blocks[i].embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_bins);
                model.tensors["quantizer.vq.layers." + std::to_string(i) + "._codebook.embed"] = model.quantizer.blocks[i].embed;
            }
        }

    }

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded    = 0;

        while(true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            read_safe(fin, n_dims);
            read_safe(fin, length);
            read_safe(fin, ftype);

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = {1, 1, 1};
            for (int i = 0; i < n_dims; i++) {
                read_safe(fin, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> buf(length);
            fin.read(&buf[0], buf.size());
            name.assign(&buf[0], buf.size());

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld, %lld], expected [%d, %d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ftype));
            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            // printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        fprintf(stderr, "%s: model size    = %7.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;
}

struct ggml_tensor * encodec_quantizer_decode_eval(
                    struct ggml_context * ctx0,
                    const encodec_model & model,
                    struct ggml_tensor  * codes) {
    // codes: [seq_length, n_codes]
    const int hidden_dim = model.hparams.hidden_dim;
    const int seq_length = codes->ne[0];
    const int n_q        = codes->ne[1];

    struct ggml_tensor * quantized_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
    quantized_out = ggml_set_zero(quantized_out);

    for (int i = 0; i < n_q; i++) {
        encodec_quant_block block = model.quantizer.blocks[i];

        struct ggml_tensor * indices   = ggml_view_1d(ctx0, codes, seq_length, i*codes->nb[1]);
        struct ggml_tensor * quantized = ggml_get_rows(ctx0, block.embed, indices);

        quantized_out = ggml_add(ctx0, quantized_out, quantized);
    }

    quantized_out = ggml_cont(ctx0, ggml_transpose(ctx0, quantized_out));

    return quantized_out;
}

struct ggml_tensor * encodec_decoder_eval(
                    struct ggml_context * ctx0,
                    const encodec_model & model,
                    struct ggml_tensor  * quantized_out) {
    const auto & hparams = model.hparams;
    const int * ratios   = hparams.ratios;
    const int stride     = hparams.stride;

    struct ggml_tensor * inpL = strided_conv_1d(
        ctx0, quantized_out, model.decoder.init_conv_w, model.decoder.init_conv_b, stride);

    // lstm
    {
        struct ggml_tensor * cur = inpL;

        const encodec_lstm lstm = model.decoder.lstm;

        // first lstm layer
        struct ggml_tensor * hs1 = forward_pass_lstm_unilayer(
            ctx0, cur, lstm.l0_ih_w, lstm.l0_hh_w, lstm.l0_ih_b, lstm.l0_hh_b);

        // second lstm layer
        struct ggml_tensor * out = forward_pass_lstm_unilayer(
            ctx0, hs1, lstm.l1_ih_w, lstm.l1_hh_w, lstm.l1_ih_b, lstm.l1_hh_b);

        inpL = ggml_add(ctx0, inpL, out);
    }

    for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
        encodec_decoder_block block = model.decoder.blocks[layer_ix];

        // upsampling layers
        inpL = ggml_elu(ctx0, inpL);

        inpL = strided_conv_transpose_1d(
            ctx0, inpL, block.us_conv_w, block.us_conv_b, ratios[layer_ix]);
        return inpL;

        struct ggml_tensor * current = inpL;

        // shortcut
        struct ggml_tensor * shortcut = strided_conv_1d(
            ctx0, inpL, block.conv_sc_w, block.conv_sc_b, stride);

        // conv1
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_1_w, block.conv_1_b, stride);

        // conv2
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_2_w, block.conv_2_b, stride);

        // residual connection
        inpL = ggml_add(ctx0, current, shortcut);
    }

    // final conv
    inpL = ggml_elu(ctx0, inpL);

    struct ggml_tensor * output = strided_conv_1d(
        ctx0, inpL, model.decoder.final_conv_w, model.decoder.final_conv_b, stride);

    return output;
}
