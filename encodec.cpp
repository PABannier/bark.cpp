#include "encodec.h"
#include "ggml.h"
#include "bark-util.h"

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <map>
#include <string>
#include <vector>

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
            model.decoder.final_conv_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

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