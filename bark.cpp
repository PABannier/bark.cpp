/* Port of Suno's Bark to C/C++. */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "bark.h"
// #include "encodec.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <string>

#define BARK_DEBUG 0
#define EPS_NORM 1e-8

#define SAMPLE_RATE 24000

#define CLS_TOKEN_ID 101
#define SEP_TOKEN_ID 102

#define TEXT_ENCODING_OFFSET 10048
#define TEXT_PAD_TOKEN 129595

#define CODEBOOK_SIZE 1024
#define N_COARSE_CODEBOOKS 2
#define N_FINE_CODEBOOKS 8

#define SEMANTIC_PAD_TOKEN 10000
#define SEMANTIC_INFER_TOKEN 129599
#define SEMANTIC_VOCAB_SIZE 10000
#define SEMANTIC_RATE_HZ 49.9

#define COARSE_RATE_HZ 75
#define COARSE_SEMANTIC_PAD_TOKEN 12048
#define COARSE_INFER_TOKEN 12050

static const size_t MB = 1024*1024;

struct bark_progress {
    float current = 0.0f;
    const char * func;

    bark_progress(const char * func): func(func) {}

    void callback(float progress) {
        float percentage = progress * 100;
        if (percentage == 0.0f) {
            fprintf(stderr, "%s: ", func);
        }
        while (percentage > current) {
            current = percentage;
            fprintf(stderr, ".");
            fflush(stderr);
            if (percentage >= 100) {
                fprintf(stderr, "\n");
            }
        }
    }
};

template<typename T>
static void read_safe(std::ifstream& fin, T& dest) {
    fin.read((char*)& dest, sizeof(T));
}

template<typename T>
static void write_safe(std::ofstream& fout, T& dest) {
    fout.write((char*)& dest, sizeof(T));
}

static void bark_print_statistics(gpt_model * model) {
    printf("\n\n");
    printf("%s: mem per token = %8.2f MB\n", __func__, model->mem_per_token/1000.0f/1000.0f);
    printf("%s:   sample time = %8.2f ms / %lld tokens\n", __func__, model->t_sample_us/1000.0f, model->n_sample);
    printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, model->t_predict_us/1000.0f, model->t_predict_us/model->n_predict/1000.0f);
    printf("%s:    total time = %8.2f ms\n", __func__, model->t_main_us/1000.0f);
    printf("\n");
}

static void softmax(std::vector<float> & logits) {
    // for numerical stability
    float maxl = -INFINITY;
    for (const auto & l : logits)
        maxl = std::max(maxl, l);

    // softmax
    float sum = 0.0;
    for (auto & l : logits) {
        l = exp(l - maxl);
        sum += l;
    }

    for (auto & l : logits)
        l /= sum;
}

static bark_token gpt_multinomial_sample(
        std::vector<float> & logits,
              std::mt19937 & rng,
                     float   temp,
                     float * eos_p) {
    int n_logits = logits.size();

    for (int i = 0; i < n_logits; ++i)
        logits[i] /= temp;

    softmax(logits);

    std::discrete_distribution<bark_token> dist(logits.begin(), logits.end());
    int next = dist(rng);

    // likelihood of EOS token
    if (eos_p)
        *eos_p = logits[logits.size() - 1];

    return next;
}

static bark_token gpt_argmax_sample(std::vector<float> & logits, float * eos_p) {
    int n_logits = logits.size();

    // testing purposes
    for (auto & l : logits) { l /= 0.7f; }

    // likelihood of EOS token
    softmax(logits);

    if (eos_p)
        *eos_p = logits[logits.size() - 1];

    int next = 0;
    float maxl = -INFINITY;

    for (int i = 0; i < n_logits; i++) {
        if (logits[i] > maxl) {
            maxl = logits[i];
            next = i;
        }
    }

    return next;
}

static bark_token gpt_sample(
            std::vector<float> & logits,
                  std::mt19937 & rng,
                         float   temp,
                         float * eos_p,
                       int64_t * t_sample_us,
                       int64_t * n_sample) {
    int64_t t_sample_start_us = ggml_time_us();

    bark_token res;
    if (temp == 0.0f) {
        res = gpt_argmax_sample(logits, eos_p);
    } else {
        res = gpt_multinomial_sample(logits, rng, temp, eos_p);
    }

    int64_t t_sample_end_us = ggml_time_us();
    *t_sample_us += (t_sample_end_us - t_sample_start_us);
    *n_sample += 1;

    return res;
}

static bool bark_vocab_load(
     const std::string & fname,
            bark_vocab * vocab,
               int32_t   expected_size) {
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

        vocab->token_to_id[word] = i;
        vocab->id_to_token[i] = word;
    }

    return true;
}

static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string strip_accents(const std::string & in_str) {
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

static void bert_tokenize(
        const bark_vocab * vocab,
              const char * text,
                 int32_t * tokens,
                 int32_t * n_tokens,
                 int32_t   n_max_tokens) {
    std::string str = text;
    std::vector<std::string> words;

    int32_t t = 0;

    auto * token_map = &vocab->token_to_id;

    // split the text into words
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

    // apply wordpiece
    for (const auto &word : words) {
        if (word.size() == 0)
            continue;

        std::string prefix = "";
        int i = 0;
        int n = word.size();

        loop:
            while (i < n) {
                if (t >= n_max_tokens - 1)
                    break;
                int j = n;
                while (j > i) {
                    auto it = token_map->find(prefix + word.substr(i, j - i));
                    if (it != token_map->end()) {
                        tokens[t++] = it->second;
                        i = j;
                        prefix = "##";
                        goto loop;
                    }
                    --j;
                }
                if (j == i) {
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                    prefix = "##";
                    ++i;
                }
            }
        }

    *n_tokens = t;
}

static void bark_tokenize_input(struct bark_context * ctx, const std::string & text) {
    auto & model = ctx->model.text_model;
    bark_vocab * vocab = &ctx->model.vocab;

    int32_t block_size = model.hparams.block_size;
    int32_t max_ctx_size = std::min(block_size, 256);
    int32_t n_tokens;

    bark_sequence tokens(max_ctx_size);
    bert_tokenize(vocab, text.data(), tokens.data(), &n_tokens, max_ctx_size);

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

    ctx->tokens = tokens;

    printf("%s: prompt: '%s'\n", __func__, text.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, ctx->tokens.size());
    for (int i = 0; i < std::min(8, (int) ctx->tokens.size()); i++) {
        printf("%d ", ctx->tokens[i]);
    }
    printf("\n");
}

static bool gpt_load_model_weights(const std::string & fname, gpt_model & model) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

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
            return 1;
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
        read_safe(fin, hparams.ftype);

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_in_vocab  = %d\n", __func__, hparams.n_in_vocab);
        printf("%s: n_out_vocab = %d\n", __func__, hparams.n_out_vocab);
        printf("%s: block_size  = %d\n", __func__, hparams.block_size);
        printf("%s: n_embd      = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head      = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer     = %d\n", __func__, hparams.n_layer);
        printf("%s: n_lm_heads  = %d\n", __func__, hparams.n_lm_heads);
        printf("%s: n_wtes      = %d\n", __func__, hparams.n_wtes);
        printf("%s: ftype       = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr       = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto & ctx = model.ctx;

    size_t buffer_size = 0;
    size_t n_tensors   = 0;

    // Evaluating context size
    {
        const auto & hparams = model.hparams;

        const int n_embd      = hparams.n_embd;
        const int n_layer     = hparams.n_layer;
        const int block_size  = hparams.block_size;
        const int n_in_vocab  = hparams.n_in_vocab;
        const int n_out_vocab = hparams.n_out_vocab;
        const int n_lm_heads  = hparams.n_lm_heads;
        const int n_wtes      = hparams.n_wtes;

        buffer_size += n_embd * ggml_type_size(GGML_TYPE_F32); // ln_f_g
        buffer_size += n_embd * ggml_type_size(GGML_TYPE_F32); // ln_f_b

        buffer_size += n_wtes * n_in_vocab * n_embd * ggml_type_size(wtype);      // wte
        buffer_size += block_size * n_embd * ggml_type_size(GGML_TYPE_F32);       // wpe
        buffer_size += n_lm_heads * n_out_vocab * n_embd * ggml_type_size(wtype); // lm_head

        buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32)); // ln_1_g
        buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32)); // ln_1_b

        buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32)); // ln_2_g
        buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32)); // ln_2_b

        buffer_size += n_layer * (3 * n_embd * n_embd * ggml_type_size(wtype));         // c_attn_attn_w
        buffer_size += n_layer * (3 * n_embd           *ggml_type_size(GGML_TYPE_F32)); // c_attn_attn_b

        buffer_size += n_layer * (n_embd * n_embd * ggml_type_size(wtype));             // c_attn_proj_w
        buffer_size += n_layer * (         n_embd * ggml_type_size(GGML_TYPE_F32));     // c_attn_proj_b

        buffer_size += n_layer * (4 * n_embd * n_embd * ggml_type_size(wtype));         // c_mlp_fc_w
        buffer_size += n_layer * (4          * n_embd * ggml_type_size(GGML_TYPE_F32)); // c_mlp_fc_b

        buffer_size += n_layer * (4 * n_embd * n_embd * ggml_type_size(wtype));         // c_mlp_proj_w
        buffer_size += n_layer * (             n_embd * ggml_type_size(GGML_TYPE_F32)); // c_mlp_proj_b

        // buffer_size += n_layer * block_size * n_embd * ggml_type_size(GGML_TYPE_F32); // memory_k
        // buffer_size += n_layer * block_size * n_embd * ggml_type_size(GGML_TYPE_F32); // memory_v

        buffer_size += 10ull*MB;    // object overhead

        n_tensors = (
            2           + // ln_f_g, ln_f_b
            n_wtes + 1  + // wte, wpe
            4 * n_layer + // ln_1_g, ln_1_b, ln_2_g, ln_2_b
            4 * n_layer + // c_attn_attn_w, c_attn_attn_b, c_attn_proj_w, c_attn_proj_b
            4 * n_layer + // c_mlp_fc_w, c_mlp_fc_b, c_mlp_proj_w, c_mlp_proj_b
            2 * n_layer + // memory_k, memory_v
            n_lm_heads    // lm_head
        );

        printf("%s: ggml tensor size = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
        printf("%s: ggml ctx size = %6.2f MB\n", __func__, buffer_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    if (!model.backend) {
        // fallback to CPU backend
        fprintf(stderr, "%s: no backend specified, using CPU backend\n", __func__);
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        fprintf(stderr, "%s: failed to initialize CPU backend\n", __func__);
        return false;
    }

    // allocate weights buffer
    model.buffer_w  = ggml_backend_alloc_buffer(model.backend, buffer_size);

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

        model.wpe    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, block_size);

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

        // create a backend buffer (can be in host or device memory)
        model.buffer_kv = ggml_backend_alloc_buffer(model.backend, memory_size + 256);

        // allocate the tensors into the backend buffer
        {
            ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer_kv);

            // this updates the pointers in the tensors to point to the correct location in the buffer
            // this is necessary since the ggml_context is .no_alloc == true
            // note that the buffer can actually be a device buffer, depending on the backend
            ggml_allocr_alloc(alloc, model.memory_k);
            ggml_allocr_alloc(alloc, model.memory_v);

            ggml_allocr_free(alloc);
        }
    }

    // load weights
    {
        ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer_w);

        size_t total_size = 0;

        std::vector<char> read_buf;

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

            auto tensor = model.tensors[name];
            ggml_set_name(tensor, name.c_str());
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

            ggml_allocr_alloc(alloc, tensor);

            if (ggml_backend_is_cpu(model.backend)) {
                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            // printf("%48s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], "float", ggml_nbytes(tensor)/1024.0/1024.0);

            total_size += ggml_nbytes(tensor);
        }

        ggml_allocr_free(alloc);
        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
        model.memsize = total_size;
    }

    fin.close();

    return true;
}

static struct ggml_cgraph * bark_build_gpt_graph(
                gpt_model * model,
              ggml_allocr * allocr,
            bark_sequence & tokens,
       std::vector<float> & logits,
                      int * n_past,
                     bool   merge_ctx,
                      int   n_threads) {
    if (!n_past) {
        fprintf(stderr, "%s: n_past is null\n", __func__);
        return NULL;
    }

    int N = tokens.size();

    const auto & hparams = model->hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.block_size;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_out_vocab;

    static size_t buf_size = ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_allocr_alloc(allocr, input);

    // avoid writing to tensors if we are only measuring the memory usage
    if (!ggml_allocr_is_measure(allocr)) {
        ggml_backend_tensor_set(input, tokens.data(), 0, N*ggml_element_size(input));
    }

    struct ggml_tensor * tok_emb;

    if (*n_past > 0) {
        assert(N == 1);
        tok_emb = ggml_get_rows(ctx0, model->wtes[0], input);
    } else {
        if (merge_ctx) {
            assert(N == 256+256+1);
            N -= 256;
        } else {
            assert(N <= n_ctx);
        }

        if (merge_ctx) {
            struct ggml_tensor * seq_embd = ggml_get_rows(ctx0, model->wtes[0], ggml_view_1d(ctx0, input, 256, 0));
            struct ggml_tensor * ctx_embd = ggml_get_rows(ctx0, model->wtes[0], ggml_view_1d(ctx0, input, 256, 256*ggml_element_size(input)));
            struct ggml_tensor * rem_embd = ggml_get_rows(ctx0, model->wtes[0], ggml_view_1d(ctx0, input,   1, 512*ggml_element_size(input)));

            struct ggml_tensor * cat_emb = ggml_add(ctx0, seq_embd, ctx_embd);

            tok_emb = ggml_new_tensor_2d(ctx0, cat_emb->type, cat_emb->ne[0], cat_emb->ne[1]+rem_embd->ne[1]);
            ggml_allocr_alloc(allocr, tok_emb);

            tok_emb = ggml_set_1d(ctx0, tok_emb, cat_emb, 0);
            tok_emb = ggml_set_1d(ctx0, tok_emb, rem_embd, cat_emb->ne[0]*cat_emb->ne[1]*ggml_element_size(cat_emb));
        } else {
            tok_emb = ggml_get_rows(ctx0, model->wtes[0], input);
        }
    }

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_allocr_alloc(allocr, position);
    if (!ggml_allocr_is_measure(allocr)) {
        for (int i = 0; i < N; ++i) {
            int32_t v = *n_past + i;
            ggml_backend_tensor_set(position, &v, i*sizeof(int32_t), sizeof(v));
        }
    }

    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(allocr, KQ_scale);
    if (!ggml_allocr_is_measure(allocr)) {
        float s = 1.0f/sqrtf(float(n_embd)/n_head);
        ggml_backend_tensor_set(KQ_scale, &s, 0, sizeof(s));
    }

    // wte + wpe
    struct ggml_tensor * inpL = ggml_add(ctx0, tok_emb, ggml_get_rows(ctx0, model->wpe, position));

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_norm(ctx0, inpL, EPS_NORM);

            // cur = ln_1_g*cur + ln_1_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ln_1_g, cur),
                        cur),
                    ggml_repeat(ctx0, model->layers[il].ln_1_b, cur));
        }

        // attn
        {
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].c_attn_attn_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model->layers[il].c_attn_attn_b, cur),
                    cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model->memory_k, N*n_embd, (ggml_element_size(model->memory_k)*n_embd)*(il*n_ctx + *n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model->memory_v, N*n_embd, (ggml_element_size(model->memory_v)*n_embd)*(il*n_ctx + *n_past));

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model->memory_k, (*n_past + N)*n_embd, il*n_ctx*ggml_element_size(model->memory_k)*n_embd),
                            n_embd/n_head, n_head, *n_past + N),
                        0, 2, 1, 3);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, KQ_scale);

            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, *n_past);

            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model->memory_v, (*n_past + N)*n_embd, il*n_ctx*ggml_element_size(model->memory_v)*n_embd),
                                n_embd/n_head, n_head, *n_past + N),
                            1, 2, 0, 3),
                        ggml_new_tensor_3d(ctx0, model->memory_v->type, *n_past + N, n_embd/n_head, n_head));

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].c_attn_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model->layers[il].c_attn_proj_b, cur),
                    cur);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, EPS_NORM);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, model->layers[il].ln_2_g, cur),
                            cur),
                        ggml_repeat(ctx0, model->layers[il].ln_2_b, cur));
            }

            // cur = fc_w*cur + fc_b
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].c_mlp_fc_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model->layers[il].c_mlp_fc_b, cur),
                    cur);

            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model->layers[il].c_mlp_proj_b, cur),
                    cur);
        }

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL, EPS_NORM);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model->ln_f_b, inpL));
    }

    inpL = ggml_mul_mat(ctx0, model->lm_heads[0], inpL);

    ggml_build_forward_expand(gf, inpL);

    // updating n_past with N (-256 if merge_ctx)
    if (n_past)
        *n_past += N;

    ggml_free(ctx0);

    return gf;
}

static bool bark_eval_text_encoder_internal(
                          struct bark_context * bctx,
                                bark_sequence & input,
                           std::vector<float> & logits,
                                          int * n_past,
                                         bool   merge_ctx,
                                          int   n_threads) {
    auto & model  = bctx->model.text_model;
    auto & allocr = bctx->allocr;

    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = bark_build_gpt_graph(
        &model, allocr, input, logits, n_past, merge_ctx, n_threads);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    return true;
}

static bool bark_eval_text_encoder(struct bark_context * bctx, int n_threads) {
    bark_sequence input = bctx->tokens;
    bark_sequence output;

    bark_progress progress( __func__);

    auto & model   = bctx->model.text_model;
    auto & hparams = model.hparams;

    const int n_vocab = hparams.n_out_vocab;

    float min_eos_p = bctx->params.min_eos_p;
    float temp      = bctx->params.temp;

    std::vector<float> logits;
    logits.resize(n_vocab);

    float eos_p = 0;
    int n_past = 0;

    for (int i = 0; i < 768; i++) {
        if (!bark_eval_text_encoder_internal(bctx, input, logits, &n_past, true, n_threads)) {
            fprintf(stderr, "%s: Could not generate token\n", __func__);
            return false;
        }

        std::vector<float> relevant_logits(logits.begin(), logits.begin() + SEMANTIC_VOCAB_SIZE);
        relevant_logits.push_back(logits[SEMANTIC_PAD_TOKEN]);

        input.clear();

        bark_token next = gpt_sample(
            logits, bctx->rng, temp, &eos_p, &model.t_sample_us, &model.n_sample);

        if (next == SEMANTIC_VOCAB_SIZE || eos_p >= min_eos_p) {
            break;
        }

        input.push_back(next);
        output.push_back(next);

        progress.callback((float) i/768);
    }

    bctx->semantic_tokens = output;

    return true;
}

static bool bark_forward_text_encoder(struct bark_context * bctx, int n_threads) {
    const int64_t t_main_start_us = ggml_time_us();

    auto & model  = bctx->model.text_model;
    auto & allocr = bctx->allocr;

    // allocate the compute buffer
    {
        // alignment required by the backend
        size_t align = ggml_backend_get_alignment(model.backend);
        bctx->allocr = ggml_allocr_new_measure(align);

        // create the graph for memory usage estimation
        std::vector<float> logits;
        struct ggml_cgraph * gf = bark_build_gpt_graph(
            &model, allocr, bctx->tokens, logits, nullptr, false, n_threads);

        // compute the required memory
        size_t mem_size = ggml_allocr_alloc_graph(bctx->allocr, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(bctx->allocr);
        bctx->buf_compute = ggml_backend_alloc_buffer(bctx->model.text_model.backend, mem_size);
        bctx->allocr = ggml_allocr_new_from_buffer(bctx->buf_compute);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size/1024.0/1024.0);
    }

    if (!bark_eval_text_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward text encoder\n", __func__);
        return false;
    }

    model.t_main_us = ggml_time_us() - t_main_start_us;

    bark_print_statistics(&model);

    // TODO: clean allocr and buf_compute in the end
    ggml_backend_buffer_free(bctx->buf_compute);

    return true;
}

static bool bark_forward_eval(
        struct bark_context * bctx,
                        int   n_threads) {
    if (!bark_forward_text_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward text encoder\n", __func__);
        return false;
    }

    return true;
}

bool bark_generate_audio(
        struct bark_context * bctx,
                std::string & text,
                std::string & dest_wav_path,
                        int   n_threads) {
    if (!bctx) {
        fprintf(stderr, "%s: invalid bark context\n", __func__);
        return false;
    }

    bark_tokenize_input(bctx, text);

    if (!bark_forward_eval(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward eval\n", __func__);
        return false;
    }

    // TODO: codes might need to get transposed...
    // TODO: call encodec API (decompress_audio)

    return true;
}

static void bark_free_model(struct gpt_model * model) {
    if (!model) {
        return;
    }

    if(model->ctx) {
        ggml_free(model->ctx);
    }

    ggml_backend_buffer_free(model->buffer_w);
    ggml_backend_free(model->backend);
}

void bark_free(struct bark_context * bctx) {
    if (!bctx) {
        return;
    }

    bark_free_model(&bctx->model.text_model);
    bark_free_model(&bctx->model.coarse_model);
    bark_free_model(&bctx->model.fine_model);

    delete bctx;
}

static struct bark_model * bark_load_model_from_file(
                         const std::string & dirname,
                         struct bark_model * model) {
    printf("%s: loading model from '%s'\n", __func__, dirname.c_str());

    // text
    {
        printf("%s: reading bark text model\n", __func__);
        const std::string fname = std::string(dirname) + "/ggml_weights_text.bin";
        if (gpt_load_model_weights(fname, model->text_model)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad text)\n", __func__, fname.c_str());
            return nullptr;
        }
    }

    // vocab
    {
        printf("%s: reading bark vocab\n", __func__);
        const std::string fname     = std::string(dirname) + "/ggml_vocab.bin";
        const gpt_hparams hparams   = model->text_model.hparams;
        const int32_t expected_size = hparams.n_in_vocab - hparams.n_out_vocab - 5;
        if (!bark_vocab_load(fname, &model->vocab, expected_size)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad text)\n", __func__, fname.c_str());
            return nullptr;
        }
    }

    // coarse
    {
        printf("\n%s: reading bark coarse model\n", __func__);
        const std::string fname = std::string(dirname) + "/ggml_weights_coarse.bin";
        if (!gpt_load_model_weights(fname, model->coarse_model)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad coarse)\n", __func__, fname.c_str());
            return nullptr;
        }
    }

    // fine
    {
        printf("\n%s: reading bark fine model\n", __func__);
        const std::string fname = std::string(dirname) + "/ggml_weights_fine.bin";
        if (!gpt_load_model_weights(fname, model->fine_model)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad fine)\n", __func__, fname.c_str());
            return nullptr;
        }
    }

    return model;
}

struct bark_context_params bark_context_default_params() {
    struct bark_context_params result = {
        /*.seed                        =*/ 0,
        /*.temp                        =*/ 0.7,
        /*.fine_temp                   =*/ 0.5,
        /*.min_eos_p                   =*/ 0.2,
        /*.sliding_window_size         =*/ 60,
        /*.max_coarse_history          =*/ 630,
    };

    return result;
}

struct bark_context * bark_load_model(const std::string & model_path) {
    int64_t t_load_start_us = ggml_time_us();

    struct bark_context * bctx = new bark_context();

    bctx->model = bark_model();
    if (!bark_load_model_from_file(model_path, &bctx->model)) {
        fprintf(stderr, "%s: failed to load model weights from '%s'\n", __func__, model_path.c_str());
        return {};
    }

    bark_context_params params = bark_context_default_params();
    bctx->rng = std::mt19937(params.seed);

    bctx->params = params;

    bctx->t_load_us = ggml_time_us() - t_load_start_us;

    return bctx;
}
