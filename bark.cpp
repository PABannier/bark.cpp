/*
Port of Suno's Bark to C/C++.

Author: Pierre-Antoine Bannier<pierreantoine.bannier@gmail.com>

Note on tokenization
--------------------
Even if bark relies on GPT to generate semantic tokens, the tokenizer is based on
Bert's multilingual cased tokenizer. This uses the WordPiece algorithm to split raw text
into tokens.

This file contains an unofficial (Google has not released an official implementation of
WordPiece) implementation of WordPiece.

Source:
https://github.com/skeskinen/bert.cpp/blob/master/bert.cpp

*/
#include "bark.h"
#include "ggml.h"
#include "util.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>

bool gpt_model_load(const std::string& fname, gpt_model& model, bark_vocab& vocab, bool has_vocab) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        read_safe(fin, hparams.n_layer);
        read_safe(fin, hparams.n_head);
        read_safe(fin, hparams.n_embd);
        read_safe(fin, hparams.block_size);
        read_safe(fin, hparams.n_in_vocab);
        read_safe(fin, hparams.n_out_vocab);
        read_safe(fin, hparams.n_lm_heads);
        read_safe(fin, hparams.n_wtes);

        printf("%s: n_in_vocab  = %d\n", __func__, hparams.n_in_vocab);
        printf("%s: n_out_vocab = %d\n", __func__, hparams.n_out_vocab);
        printf("%s: block_size  = %d\n", __func__, hparams.block_size);
        printf("%s: n_embd      = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head      = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer     = %d\n", __func__, hparams.n_layer);
        printf("%s: n_lm_heads  = %d\n", __func__, hparams.n_lm_heads);
        printf("%s: n_wtes      = %d\n", __func__, hparams.n_wtes);
    }

    if (has_vocab) {
        int32_t n_vocab;
        read_safe(fin, n_vocab);

        // 5 special tokens: [UNK, SEP, MASK, PAD, CLS]
        if (n_vocab != model.hparams.n_in_vocab - model.hparams.n_out_vocab - 5) {
            fprintf(stderr, "%s: wrong voculary size (%d != %d)\n", __func__, n_vocab, model.hparams.n_in_vocab);
            return false;
        }

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(fin, len);

            if (len > 0) {
                tmp.resize(len);
                fin.read(&tmp[0], tmp.size()); // read to buffer
                word.assign(&tmp[0], tmp.size());
            } else {
                word = "";
            }

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    // if (wtype == GGML_TYPE_COUNT) {
    //     fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
    //             __func__, fname.c_str(), model.hparams.ftype);
    //     return false;
    // }
    ggml_type wtype = GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd      = hparams.n_embd;
        const int n_layer     = hparams.n_layer;
        const int block_size  = hparams.block_size;
        const int n_in_vocab  = hparams.n_in_vocab;
        const int n_out_vocab = hparams.n_out_vocab;
        const int n_lm_heads  = hparams.n_lm_heads;
        const int n_wtes      = hparams.n_wtes;

        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_wtes*n_in_vocab*n_embd*ggml_type_sizef(wtype);     // wte
        ctx_size += block_size*n_embd*ggml_type_sizef(GGML_TYPE_F32); // wpe
        ctx_size += n_lm_heads*n_out_vocab*n_embd*ggml_type_sizef(wtype); // lm_head

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_b

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_b

        ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_sizef(wtype));         // c_attn_attn_w
        ctx_size += n_layer*(       3*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_attn_attn_b

        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype));           // c_attn_proj_w
        ctx_size += n_layer*(       n_embd*ggml_type_sizef(GGML_TYPE_F32));   // c_attn_proj_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_fc_w
        ctx_size += n_layer*(       4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_proj_w
        ctx_size += n_layer*(         n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b

        ctx_size += block_size*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += block_size*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (6 + 12*n_layer)*512; // object overhead

        printf("%s: ggml tensor size = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd     = hparams.n_embd;
        const int n_layer    = hparams.n_layer;
        const int block_size = hparams.block_size;
        const int n_in_vocab  = hparams.n_in_vocab;
        const int n_out_vocab = hparams.n_out_vocab;
        const int n_lm_heads = hparams.n_lm_heads;
        const int n_wtes     = hparams.n_wtes;

        model.layers.resize(n_layer);
        model.lm_heads.resize(n_lm_heads);
        model.wtes.resize(n_wtes);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.wpe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, block_size);

        for (int i = 0; i < n_wtes; i++) {
            model.wtes[i] = ggml_new_tensor_2d(ctx, wtype, n_embd, n_in_vocab);
            model.tensors["model/wte/" + std::to_string(i)] = model.wtes[i];
        }

        for (int i = 0; i < n_lm_heads; i++) {
            model.lm_heads[i] = ggml_new_tensor_2d(ctx, wtype, n_embd, n_out_vocab);
            model.tensors["model/lm_head/" + std::to_string(i)] = model.lm_heads[i];
        }

        model.tensors["model/ln_f/g"] = model.ln_f_g;
        model.tensors["model/ln_f/b"] = model.ln_f_b;

        model.tensors["model/wpe"]     = model.wpe;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_1_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.ln_2_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_2_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, n_embd);
            layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_mlp_fc_w    = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w  = ggml_new_tensor_2d(ctx, wtype,         4*n_embd, n_embd);
            layer.c_mlp_proj_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"]        = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"]        = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"]        = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"]        = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"]    = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"]    = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"]  = layer.c_mlp_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"]  = layer.c_mlp_proj_b;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd     = hparams.n_embd;
        const int n_layer    = hparams.n_layer;
        const int block_size = hparams.block_size;

        const int n_mem      = n_layer*block_size;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        size_t total_size = 0;

        while(true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(fin, n_dims);
            read_safe(fin, length);
            read_safe(fin, ttype);

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(fin, ne[i]);
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            // printf("%48s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], "float", ggml_nbytes(tensor)/1024.0/1024.0);

            total_size += ggml_nbytes(tensor);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
        model.memsize = total_size;
    }

    fin.close();

    return true;
}

bool bark_model_load(std::string & dirname, bark_model & model) {
    printf("%s: loading model from '%s'\n", __func__, dirname.c_str());

    // text
    {
        printf("%s: reading bark text model\n", __func__);
        const std::string fname = dirname + "/ggml_weights_text.bin";
        if(!gpt_model_load(fname, model.text_model, model.vocab, true)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad text)\n", __func__, fname.c_str());
            return false;
        }
        model.memsize += model.text_model.memsize;
    }

    // coarse
    {
        printf("\n%s: reading bark coarse model\n", __func__);
        const std::string fname = dirname + "/ggml_weights_coarse.bin";
        if(!gpt_model_load(fname, model.coarse_model, model.vocab, false)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad coarse)\n", __func__, fname.c_str());
            return false;
        }
        model.memsize += model.coarse_model.memsize;
    }

    // fine
    {
        printf("\n%s: reading bark fine model\n", __func__);
        const std::string fname = dirname + "/ggml_weights_fine.bin";
        if(!gpt_model_load(fname, model.fine_model, model.vocab, false)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad fine)\n", __func__, fname.c_str());
            return false;
        }
        model.memsize += model.fine_model.memsize;
    }

    // codec
    {
        printf("\n%s: reading bark codec model\n", __func__);
        const std::string fname = dirname + "/ggml_weights_codec.bin";
        if(!encodec_model_load(fname, model.codec_model)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad codec)\n", __func__, fname.c_str());
            return false;
        }
        model.memsize += model.coarse_model.memsize;
    }

    printf("\n%s: total model size  = %8.2f MB\n", __func__, model.memsize/1024.0/1024.0);

    return true;
}

std::string bert_normalize_prompt(const std::string &text) {
    std::string text2 = strip_accents(text);
    for (size_t i = 0; i < text2.size(); i += utf8_len(text2[i])) {
        char c = text2[i];
        if (c >= 'A' && c <= 'Z')
            text2[i] = c - 'A' + 'a';
    }
    return text2;
}

void bert_tokenize(
    const bark_vocab& vocab,
    const char * text,
    int32_t * tokens,
    int32_t * n_tokens,
    int32_t n_max_tokens) {
    std::string str = text;
    std::vector<std::string> words;

    // first split the text into words
    {
        str = bert_normalize_prompt(str);

        std::string pat = R"([[:punct:]]|[[:alpha:]]+|[[:digit:]]+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (std::string x : m)
                words.push_back(x);
            str = m.suffix();
        }
    }

    int32_t t = 0;
    tokens[t++] = CLS_TOKEN_ID;

    // find the longest tokens that form the words:
    for (const auto &word : words) {
        if (word.size() == 0)
            continue;

        int i = 0;
        int n = word.size();
        auto *token_map = &vocab.token_to_id;
    loop:
        while (i < n) {
            if (t >= n_max_tokens - 1)
                break;

            int j = n;
            while (j > i) {
                auto it = token_map->find(word.substr(i, j - i));
                if (it != token_map->end()) {
                    tokens[t++] = it->second;
                    i = j;
                    token_map = &vocab.subword_token_to_id;
                    goto loop;
                }
                --j;
            }
            if (j == i) {
                fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                token_map = &vocab.subword_token_to_id;
                ++i;
            }
        }
    }
    tokens[t++] = SEP_TOKEN_ID;
    *n_tokens = t;
}

void bark_generate_audio(bark_model model, const bark_vocab& vocab, const char * text) {
    // tokenize text (bert tokenizer)
    int32_t max_ctx_size = model.text_model.hparams.block_size;
    int32_t n_tokens;

    std::vector<bark_vocab::id> tokens;
    tokens.resize(max_ctx_size);
    bert_tokenize(vocab, text, tokens.data(), &n_tokens, max_ctx_size);

    printf("\ntokens: ");
    for (int i = 0; i < n_tokens; i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    // encode text

    // generate audio (encodec)
}

int main(int argc, char **argv) {
    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_load_us = 0;
    int64_t t_eval_us = 0;

    bark_model model;
    std::string fname = "./ggml_weights";

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if(!bark_model_load(fname, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // forward pass
    const std::string prompt = "This is an audio";
    {
        const int64_t t_eval_us_start = ggml_time_us();

        // call to generate audio
        bark_generate_audio(model, model.vocab, prompt.data());

        t_eval_us = ggml_time_us() - t_eval_us_start;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    // TODO: write wrapper
    ggml_free(model.coarse_model.ctx);
    ggml_free(model.fine_model.ctx);
    ggml_free(model.text_model.ctx);
    ggml_free(model.codec_model.ctx);

    return 0;
}