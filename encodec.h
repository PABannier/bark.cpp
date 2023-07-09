/* This is a shortened version of the original Encodec.CPP here: https://github.com/PABannier/encodec.cpp.
Since bark only uses the decoder, only the decoding forward pass is present in this file.
*/
#pragma once

#include "ggml.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <string>
#include <vector>

struct encodec_hparams {
    int32_t in_channels          = 1;
    int32_t hidden_dim           = 128;
    int32_t n_filters            = 32;
    int32_t ratios[4]            = {8, 5, 4, 2};
    int32_t kernel_size          = 7;
    int32_t residual_kernel_size = 3;
    int32_t compress             = 2;
    int32_t n_lstm_layers        = 2;
    int32_t stride               = 1;

    // 24kbps (n_q=32)
    int32_t n_q                  = 32;
    int32_t n_bins               = 1024;
    int32_t sr                   = 24000;
};

// res + downsample block at some ratio
struct encodec_encoder_block {
    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;

    // downsampling layers
    struct ggml_tensor * ds_conv_w;
    struct ggml_tensor * ds_conv_b;
};

struct encodec_lstm {
    struct ggml_tensor * l0_ih_w;
    struct ggml_tensor * l0_hh_w;

    struct ggml_tensor * l0_ih_b;
    struct ggml_tensor * l0_hh_b;

    struct ggml_tensor * l1_ih_w;
    struct ggml_tensor * l1_hh_w;

    struct ggml_tensor * l1_ih_b;
    struct ggml_tensor * l1_hh_b;
};

struct encodec_quant_block {
    struct ggml_tensor * embed;
};

struct encodec_quantizer {
    std::vector<encodec_quant_block> blocks;
};

struct encodec_decoder_block {
    //upsampling layers
    struct ggml_tensor * us_conv_w;
    struct ggml_tensor * us_conv_b;

    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;
};

struct encodec_decoder {
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    std::vector<encodec_decoder_block> blocks;
};

struct encodec_model {
    encodec_hparams hparams;

    encodec_quantizer quantizer;
    encodec_decoder   decoder;

    // context
    struct ggml_context * ctx;
    int n_loaded;

    std::map<std::string, struct ggml_tensor *> tensors;
};


bool encodec_model_load(const std::string& fname, encodec_model& model);