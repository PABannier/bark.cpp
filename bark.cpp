/*
Port of Suno's Bark to C/C++.

Author: Pierre-Antoine Bannier <pierreantoine.bannier@gmail.com>

Note on tokenization
--------------------
Even if bark relies on GPT to generate semantic tokens, the tokenizer is based on
Bert's multilingual cased tokenizer. This uses the WordPiece algorithm to split raw text
into tokens.

This file contains an unofficial (Google has not released an official implementation of
WordPiece) implementation of WordPiece.

Source:
https://github.com/skeskinen/bert.cpp/
*/
#include "bark.h"
#include "ggml.h"
#include "bark-util.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <string>

bool bark_vocab_load(const std::string& fname, bark_vocab& vocab, int32_t expected_size) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: faield to open '%s'\n", __func__, fname.c_str());
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

    int32_t n_vocab;
    read_safe(fin, n_vocab);

    // 5 special tokens: [UNK, SEP, MASK, PAD, CLS]
    if (n_vocab != expected_size) {
        fprintf(stderr, "%s: wrong voculary size (%d != %d)\n", __func__, n_vocab, expected_size);
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

    return true;
}

bool gpt_model_load(const std::string& fname, gpt_model& model) {
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

        const int n_embd      = hparams.n_embd;
        const int n_layer     = hparams.n_layer;
        const int block_size  = hparams.block_size;
        const int n_in_vocab  = hparams.n_in_vocab;
        const int n_out_vocab = hparams.n_out_vocab;
        const int n_lm_heads  = hparams.n_lm_heads;
        const int n_wtes      = hparams.n_wtes;

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

bool bark_model_load(const std::string & dirname, bark_model & model) {
    printf("%s: loading model from '%s'\n", __func__, dirname.c_str());

    // text
    {
        printf("%s: reading bark text model\n", __func__);
        const std::string fname = dirname + "/ggml_weights_text.bin";
        if(!gpt_model_load(fname, model.text_model)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad text)\n", __func__, fname.c_str());
            return false;
        }
        model.memsize += model.text_model.memsize;
    }

    // vocab
    {
        printf("%s: reading bark vocab\n", __func__);
        const std::string fname     = dirname + "/ggml_vocab.bin";
        const gpt_hparams hparams   = model.text_model.hparams;
        const int32_t expected_size = hparams.n_in_vocab - hparams.n_out_vocab - 5;
        if(!bark_vocab_load(fname, model.vocab, expected_size)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad text)\n", __func__, fname.c_str());
            return false;
        }
    }

    // coarse
    {
        printf("\n%s: reading bark coarse model\n", __func__);
        const std::string fname = dirname + "/ggml_weights_coarse.bin";
        if(!gpt_model_load(fname, model.coarse_model)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad coarse)\n", __func__, fname.c_str());
            return false;
        }
        model.memsize += model.coarse_model.memsize;
    }

    // fine
    {
        printf("\n%s: reading bark fine model\n", __func__);
        const std::string fname = dirname + "/ggml_weights_fine.bin";
        if(!gpt_model_load(fname, model.fine_model)) {
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


std::string strip_accents(const std::string &in_str) {
    std::string out_str;
    std::map<std::string, char> accent_map = {{"À", 'A'},{"Á", 'A'},
        {"Â", 'A'},{"Ã", 'A'},{"Ä", 'A'},{"Å", 'A'},{"à", 'a'},{"á", 'a'},
        {"â", 'a'},{"ã", 'a'},{"ä", 'a'},{"å", 'a'},{"È", 'E'},{"É", 'E'},
        {"Ê", 'E'},{"Ë", 'E'},{"è", 'e'},{"é", 'e'},{"ê", 'e'},{"ë", 'e'},
        {"Ì", 'I'},{"Í", 'I'},{"Î", 'I'},{"Ï", 'I'},{"ì", 'i'},{"í", 'i'},
        {"î", 'i'},{"ï", 'i'},{"Ò", 'O'},{"Ó", 'O'},{"Ô", 'O'},{"Õ", 'O'},
        {"Ö", 'O'},{"ò", 'o'},{"ó", 'o'},{"ô", 'o'},{"õ", 'o'},{"ö", 'o'},
        {"Ù", 'U'},{"Ú", 'U'},{"Û", 'U'},{"Ü", 'U'},{"ù", 'u'},{"ú", 'u'},
        {"û", 'u'},{"ü", 'u'},{"Ý", 'Y'},{"ý", 'y'},{"Ç", 'C'},{"ç", 'c'},
        {"Ñ", 'N'},{"ñ", 'n'},
    };

    for (size_t i = 0; i < in_str.length();) {
        int len = utf8_len(in_str[i]);
        std::string cur = in_str.substr(i, len);
        auto iter = accent_map.find(cur);
        if (iter != accent_map.end())
            out_str += iter->second;
        else
            out_str += cur;

        i += len;
    }

    return out_str;
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
        str = strip_accents(text);

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

    // find the longest tokens that form the words:
    for (const auto &word : words)
    {
        if (word.size() == 0)
            continue;

        int i = 0;
        int n = word.size();
        auto *token_map = &vocab.token_to_id;
    loop:
        while (i < n)
        {
            if (t >= n_max_tokens - 1)
                break;
            int j = n;
            while (j > i)
            {
                auto it = token_map->find(word.substr(i, j - i));
                if (it != token_map->end())
                {
                    tokens[t++] = it->second;
                    i = j;
                    token_map = &vocab.subword_token_to_id;
                    goto loop;
                }
                --j;
            }
            if (j == i)
            {
                fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                token_map = &vocab.subword_token_to_id;
                ++i;
            }
        }
    }
    *n_tokens = t;
}

bool fine_gpt_eval(
        const gpt_model & model,
        const int n_threads,
        const int codebook_ix,
        const std::vector<std::vector<bark_vocab::id>> & embd_inp,
              std::vector<std::vector<float>>          & logits,
              size_t                                   & mem_per_token) {
    // embd_inp: (n_channels, seq_length)
    const int n_past = 0;
    const int n_codes = embd_inp.size();
    const int N = embd_inp[0].size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.block_size;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_out_vocab;

    const int n_codes_given = hparams.n_codes_given;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N*n_codes > buf_size) {
        const size_t buf_size_new = 1.2*(mem_per_token*N*n_codes); // add 20% to account for ggml object overhead

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, N, n_codes);
    for (int c = 0; c < n_codes; c++) {
        int offset = ggml_element_size(input)*c*N;
        memcpy((void *) ((char *) input->data + offset), embd_inp[c].data(), N*ggml_element_size(input));
    }

    struct ggml_tensor * embd = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, N, codebook_ix+1);
    struct ggml_tensor * curr;

    for (int wte_ix = 0; wte_ix < codebook_ix+1; wte_ix++) {
        curr = ggml_get_rows(ctx0,
                    model.wtes[wte_ix],
                    ggml_view_1d(ctx0, input, N, N*wte_ix*ggml_element_size(input)));

        embd = ggml_set_2d(ctx0, embd, curr,
                    ggml_element_size(curr)*n_embd,
                    ggml_element_size(curr)*wte_ix*N*n_embd);
    }

    // [n_embd, N, codebook_ix+1] -> [codebook_ix+1, n_embd, N]
    embd = ggml_cont(ctx0, ggml_permute(ctx0, embd, 1, 2, 0, 3));
    // // [1, n_embd, N]
    embd = ggml_sum_rows(ctx0, embd);
    // [N, n_embd]
    embd = ggml_cont(ctx0, ggml_permute(ctx0, embd, 2, 0, 1, 3));

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    for (int i = 0; i < N; ++i) {
        ((int32_t *) position->data)[i] = i;
    }
    struct ggml_tensor * pos_embd = ggml_get_rows(ctx0, model.wpe, position);

    // wte + wpe
    struct ggml_tensor * inpL = ggml_add(ctx0, embd, pos_embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ln_1_g, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_attn_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_attn_b, cur),
                    cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0,
            //                ggml_reshape_3d(ctx0,
            //                    ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                        ggml_new_tensor_3d(ctx0, model.memory_v->type, n_past + N, n_embd/n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_proj_b, cur),
                    cur);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, model.layers[il].ln_2_g, cur),
                            cur),
                        ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_fc_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur),
                    cur);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(ctx0, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur),
                    cur);
        }

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // inpL = WTE * inpL
    // [ 1024, 1056] - model.lm_head
    // [ 1024, N]     - inpL
    inpL = ggml_mul_mat(ctx0, model.lm_heads[codebook_ix - n_codes_given], inpL);

    // run the computation
    // ggml_build_forward_expand(&gf, toy);
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    // [1056, 1024]
    logits.resize(N);

    for (int i = 0; i < N; i++) {
        std::vector<float> tmp(n_vocab);
        memcpy(tmp.data(), (float *) ggml_get_data(inpL) + (i*n_vocab), sizeof(float)*n_vocab);
        logits[i] = tmp;
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}


bool gpt_eval(
        const gpt_model & model,
        const int n_threads,
        const int n_past,
        const bool merge_ctx,
        const std::vector<bark_vocab::id> & embd_inp,
              std::vector<float>          & embd_w,
              size_t                      & mem_per_token) {
    int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.block_size;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_out_vocab;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.2*(mem_per_token*N); // add 20% to account for ggml object overhead

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(input->data, embd_inp.data(), N*ggml_element_size(input));

    struct ggml_tensor * embd;

    if (!merge_ctx) {
        // usually only one token is in the sequence (since the context is saved in
        // memory_k and memory_v)
        embd = ggml_get_rows(ctx0, model.wtes[0], input);
    } else {
        // first step (context merging)
        struct ggml_tensor * seq_embd = ggml_get_rows(ctx0, model.wtes[0], ggml_view_1d(ctx0, input, 256, 0));
        struct ggml_tensor * ctx_embd = ggml_get_rows(ctx0, model.wtes[0], ggml_view_1d(ctx0, input, 256, 256*ggml_element_size(input)));
        struct ggml_tensor * rem_embd = ggml_get_rows(ctx0, model.wtes[0], ggml_view_1d(ctx0, input,   1, 512*ggml_element_size(input)));

        struct ggml_tensor * merged_embd = ggml_add(ctx0, seq_embd, ctx_embd);

        embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, merged_embd->ne[0], merged_embd->ne[1]+rem_embd->ne[1]);
        embd = ggml_set_1d(ctx0, embd, merged_embd, 0);
        embd = ggml_set_1d(ctx0, embd, rem_embd, merged_embd->ne[0]*merged_embd->ne[1]*ggml_element_size(merged_embd));

        N -= 256;  // merge context, input size is reduced
    }

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    for (int i = 0; i < N; ++i) {
        ((int32_t *) position->data)[i] = n_past + i;
    }

    // wte + wpe
    struct ggml_tensor * inpL = ggml_add(ctx0,
        embd, ggml_get_rows(ctx0, model.wpe, position));

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ln_1_g, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_attn_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_attn_b, cur),
                    cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0,
            //                ggml_reshape_3d(ctx0,
            //                    ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                        ggml_new_tensor_3d(ctx0, model.memory_v->type, n_past + N, n_embd/n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_proj_b, cur),
                    cur);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, model.layers[il].ln_2_g, cur),
                            cur),
                        ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_fc_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur),
                    cur);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(ctx0, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur),
                    cur);
        }

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(ctx0, model.lm_heads[0], inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    // return result just for the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }

    ggml_free(ctx0);

    return true;
}

bark_vocab::id gpt_sample(
        const std::vector<float>& logits,
        double temp,
        std::mt19937 & rng,
        float * eos_p) {
    int n_logits = logits.size();

    std::vector<std::pair<double, bark_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            logits_id.push_back(std::make_pair(logits[i]*scale, i));
        }
    }

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    // likelihood of EOS token
    if (eos_p)
        *eos_p = probs.back();

    return logits_id[idx].second;
}

std::vector<bark_vocab::id> bark_forward_text_encoder(
    const std::vector<bark_vocab::id> & tokens,
    const gpt_model model,
    const int n_threads,
    const float temp,
    const bool early_stop,
    const float min_eos_p) {

    std::vector<bark_vocab::id> out;
    int n_past = 0;
    float eos_p = 0;

    std::vector<bark_vocab::id> input = tokens;
    std::vector<float> logits;

    std::mt19937 rng(0);

    // dry run to estimate mem_per_token
    size_t mem_per_token = 0;
    gpt_eval(model, n_threads, 0, false, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = 0; i < 768; i++) {
        const bool merge_ctx = i == 0;
        gpt_eval(model, n_threads, n_past, merge_ctx, input, logits, mem_per_token);

        float logits_pad_token = logits[SEMANTIC_PAD_TOKEN];
        logits.resize(SEMANTIC_VOCAB_SIZE);

        if (early_stop)
            logits.push_back(logits[logits_pad_token]);

        n_past += input.size();
        if (i == 0)
            n_past -= 256;  // first step, context are merged

        input.clear();

        bark_vocab::id sampled_id = gpt_sample(logits, temp, rng, &eos_p);

        input.push_back(sampled_id);
        out.push_back(sampled_id);

        printf("%d ", sampled_id);
        fflush(stdout);

        if (early_stop && ((sampled_id == SEMANTIC_VOCAB_SIZE) || (eos_p > min_eos_p)))
            break;
    }

    printf("\n\nsemantic sequence length: %zu\n\n", out.size());

    return out;
}

bool bark_generate_audio(
        bark_model model,
        const bark_vocab& vocab,
        const char * text,
        const int n_threads) {
    std::vector<bark_vocab::id> tokens;

    // TODO move into params
    // const int top_k = 10;
    const int seed  = 0;

    // const float top_p     = 0.2;
    const float temp      = 1.0;
    const float fine_temp = 0.5;

    const bool early_stop = true;

    const int sliding_window_size = 60;
    const int max_coarse_history = 630;

    const float min_eos_p = 0.2;

    std::mt19937 rng(seed);

    // tokenize text (bert tokenizer)
    {
        // max bark length: 256
        int32_t max_ctx_size = std::min(model.text_model.hparams.block_size, 256);
        int32_t n_tokens;
        tokens.resize(max_ctx_size);

        bert_tokenize(vocab, text, tokens.data(), &n_tokens, max_ctx_size);
        for (int i = 0; i < (int) tokens.size(); i++)
            tokens[i] += TEXT_ENCODING_OFFSET;

        if (n_tokens < max_ctx_size) {
            for (int i = n_tokens; i < max_ctx_size; i++)
                tokens[i] = TEXT_PAD_TOKEN;
        } else if (n_tokens > max_ctx_size) {
            fprintf(stderr, "%s: input sequence is too long (%d > 256), truncating sequence", __func__, n_tokens);
        }

        tokens.resize(max_ctx_size);

        // semantic history
        for (int i = 0; i < 256; i++)
            tokens.push_back(SEMANTIC_PAD_TOKEN);

        tokens.push_back(SEMANTIC_INFER_TOKEN);

        assert(tokens.size() == 256 + 256 + 1);
    }

    printf("%s: prompt: '%s'\n", __func__, text);
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, tokens.size());
    for (int i = 0; i < std::min(8, (int) tokens.size()); i++) {
        printf("%d ", tokens[i]);
    }

    printf("\n\n");

    // encode text (text model)
    std::vector<bark_vocab::id> out_semantic = bark_forward_text_encoder(
        tokens, model.text_model, n_threads, temp, early_stop, min_eos_p);

    // coarse encoding (coarse model)
    std::vector<std::vector<bark_vocab::id>> out_coarse(N_COARSE_CODEBOOKS);
    {
        std::vector<bark_vocab::id> out;
        float semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS;
        int max_semantic_history = floorf(max_coarse_history / semantic_to_coarse_ratio);

        int n_steps = floorf(out_semantic.size() * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS) * N_COARSE_CODEBOOKS;
        int step_ix = 0;

        BARK_ASSERT(n_steps > 0);
        BARK_ASSERT(n_steps % N_COARSE_CODEBOOKS == 0);

        int n_window_steps = ceilf(static_cast<float>(n_steps) / sliding_window_size);

        std::vector<bark_vocab::id> input = out_semantic;
        std::vector<float> logits;

        // dry run to estimate mem_per_token
        size_t mem_per_token = 0;
        gpt_eval(model.coarse_model, n_threads, 0, false, { 0, 1, 2, 3 }, logits, mem_per_token);

        for(int i = 0; i < n_window_steps; i++) {
            int semantic_ix = roundf(n_steps / semantic_to_coarse_ratio);

            std::vector<bark_vocab::id> input_in(input.begin() + std::max(semantic_ix-max_semantic_history, 0), input.end());
            size_t original_size = input_in.size();
            input_in.resize(256);

            // padding from the right side
            for (int ix = original_size; ix < 256; ix++)
                input_in[ix] = COARSE_SEMANTIC_PAD_TOKEN;

            input_in.push_back(COARSE_INFER_TOKEN);

            // concatenate input_in and input_coarse
            input_in.insert(
                input_in.end(),
                std::make_move_iterator(out.end() - std::min(max_coarse_history, (int) out.size())),
                std::make_move_iterator(out.end())
            );

            int n_past = 0;
            mem_per_token *= 1.1;  // context length is growing, mem_per_token must grow as well

            for (int j = 0; j < sliding_window_size; j++) {
                if (step_ix >= n_steps)
                    continue;

                gpt_eval(model.coarse_model, n_threads, n_past, false, input_in, logits, mem_per_token);

                n_past += input_in.size();
                input_in.clear();

                bool is_major = step_ix % N_COARSE_CODEBOOKS == 0;
                int start_ix  = SEMANTIC_VOCAB_SIZE + (1 - is_major) * CODEBOOK_SIZE;
                int end_ix    = SEMANTIC_VOCAB_SIZE + (2 - is_major) * CODEBOOK_SIZE;
                std::vector<float> relevant_logits(logits.begin() + start_ix, logits.begin() + end_ix);

                bark_vocab::id sampled_id = gpt_sample(relevant_logits, temp, rng, NULL);
                sampled_id += start_ix;

                input_in.push_back(sampled_id);
                out.push_back(sampled_id);

                printf("%d ", sampled_id);
                fflush(stdout);

                step_ix += 1;
            }
        }

        BARK_ASSERT((int) out.size() == n_steps);
        BARK_ASSERT(out.size() % N_COARSE_CODEBOOKS == 0);

        // gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
        int mid = out.size() / 2;
        for(int ix = 0; ix < (int) out.size(); ix++) {
            if(ix < mid)
                out_coarse[0].push_back(out[ix] - SEMANTIC_VOCAB_SIZE);
            else
                // !! this only works when CODEBOOK_SIZE = 2
                out_coarse[1].push_back(out[ix] - SEMANTIC_VOCAB_SIZE - CODEBOOK_SIZE);
        }

        printf("\n\ncoarse sequence length: %zu\n\n", out.size());
    }

    // fine encoding (fine model)
    std::vector<std::vector<bark_vocab::id>> out_fine;
    {
        std::vector<std::vector<bark_vocab::id>> input = out_coarse;
        std::vector<std::vector<float>> logits;

        size_t mem_per_token = 0;

        int n_coarse          = input.size();
        int original_seq_len  = input[0].size();
        int n_remove_from_end = 0;

        // channel padding
        for(int i = N_COARSE_CODEBOOKS; i < N_FINE_CODEBOOKS; i++) {
            std::vector<bark_vocab::id> tmp(original_seq_len, CODEBOOK_SIZE);
            input.push_back(tmp);
        }

        // spatial padding if sequence is too short
        if(original_seq_len < 1024) {
            n_remove_from_end = 1024 - original_seq_len;
            for (int i = 0; i < (int) input.size(); i++) {
                for (int j = original_seq_len; j < 1024; j++) {
                    input[i].push_back(CODEBOOK_SIZE);
                }
            }
        }

        fine_gpt_eval(model.fine_model, n_threads, n_coarse, { {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8} }, logits, mem_per_token);

        int n_loops = std::max(0, (int) ceilf((input[0].size() - 1024)/512.f)) + 1;

        std::vector<std::vector<bark_vocab::id>> in_arr = input;

        for (int n = 0; n < n_loops; n++) {
            int start_ix          = std::min(n * 512, (int) in_arr[0].size() - 1024);
            int start_fill_ix     = std::min(n * 512, (int) in_arr[0].size() - 512);
            int rel_start_fill_ix = start_fill_ix - start_ix;

            std::vector<std::vector<bark_vocab::id>> in_buffer(in_arr.size());
            for (int ix = 0; ix < (int) in_buffer.size(); ix++) {
                std::vector<bark_vocab::id> buf(in_arr[ix].begin() + start_ix, in_arr[ix].begin() + start_ix + 1024);
                in_buffer[ix] = buf;
            }

            for (int nn = n_coarse; nn < N_FINE_CODEBOOKS; nn++) {
                fine_gpt_eval(model.fine_model, n_threads, nn, in_buffer, logits, mem_per_token);

                std::vector<bark_vocab::id> predictions(CODEBOOK_SIZE - rel_start_fill_ix);

                for (int i = 0; i < (int) logits.size(); i++) {
                    std::vector<float> relevant_logits = logits[i];
                    relevant_logits.resize(CODEBOOK_SIZE);

                    bark_vocab::id sampled_id = gpt_sample(relevant_logits, fine_temp, rng, NULL);
                    in_buffer[nn][rel_start_fill_ix+i] = sampled_id;
                }
            }

            // transfer over info into model_in
            for (int nn = n_coarse; nn < N_FINE_CODEBOOKS; nn++) {
                for (int j = 0; j < CODEBOOK_SIZE - rel_start_fill_ix; j++) {
                    in_arr[nn][start_fill_ix+j] = in_buffer[nn][rel_start_fill_ix+j];
                }
            }
        }

        if (n_remove_from_end > 0) {
            for (int i = 0; i < (int) in_arr.size(); i++) {
                in_arr[i].resize(in_arr[i].size() - n_remove_from_end);
            }
        }

        BARK_ASSERT(out_coarse[0].size() == in_arr[0].size());

        out_fine = in_arr;
    }

    return true;
}
