#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "bark.h"
#include "encodec.h"

#define BARK_MAX_NODES 4096

#define EPS_NORM 1e-5f

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static const size_t MB = 1024 * 1024;

typedef int32_t bark_token;
typedef std::vector<int32_t> bark_sequence;
typedef std::vector<std::vector<int32_t>> bark_codes;

struct bark_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

struct gpt_hparams {
    int32_t n_in_vocab;
    int32_t n_out_vocab;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_embd;
    int32_t block_size;
    int32_t n_lm_heads;
    int32_t n_wtes;
    int32_t ftype;
    int32_t bias;

    int32_t n_codes_given = 1;
};

struct gpt_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // mlp
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt_model {
    gpt_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wpe;                   //  position embedding
    std::vector<struct ggml_tensor*> wtes;      //     token embedding
    std::vector<struct ggml_tensor*> lm_heads;  // language model head

    std::vector<gpt_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx_w;
    struct ggml_context * ctx_kv;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;
    ggml_backend_buffer_t buffer_kv;

    std::map<std::string, struct ggml_tensor*> tensors;

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;
    int64_t t_main_us = 0;

    int64_t n_sample = 0;

    int64_t memsize = 0;
};

struct bark_model {
    // The token encoders
    struct gpt_model semantic_model;
    struct gpt_model coarse_model;
    struct gpt_model fine_model;

    // The vocabulary for the semantic encoder
    struct bark_vocab vocab;
};

struct bark_context {
    struct bark_model text_model;

    struct encodec_context * encodec_ctx;

    // buffer for model evaluation
    ggml_backend_buffer_t buf_compute;

    // custom allocator
    ggml_gallocr_t allocr = NULL;
    int n_gpu_layers = 0;

    std::mt19937 rng;

    bark_sequence tokens;
    bark_sequence semantic_tokens;

    bark_codes coarse_tokens;
    bark_codes fine_tokens;

    float * generated_audio = NULL;
    int n_generated_samples = 0;

    // hyperparameters
    bark_context_params params;

    // encodec parameters
    std::string encodec_model_path;

    // statistics
    bark_statistics stats;
};

template <typename T>
static void read_safe(std::ifstream& fin, T& dest) {
    fin.read((char*)&dest, sizeof(T));
}

template <typename T>
static void write_safe(std::ofstream& fout, T& dest) {
    fout.write((char*)&dest, sizeof(T));
}

static void bark_print_statistics(gpt_model* model) {
    printf("\n\n");
    printf("%s:   sample time = %8.2f ms / %lld tokens\n", __func__, model->t_sample_us / 1000.0f, model->n_sample);
    printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, model->t_predict_us / 1000.0f, model->t_predict_us / model->n_sample / 1000.0f);
    printf("%s:    total time = %8.2f ms\n", __func__, model->t_main_us / 1000.0f);
    printf("\n");
}

static void softmax(std::vector<float>& logits) {
    // for numerical stability
    float maxl = -INFINITY;
    for (const auto& l : logits)
        maxl = std::max(maxl, l);

    // softmax
    float sum = 0.0;
    for (auto& l : logits) {
        l = exp(l - maxl);
        sum += l;
    }

    for (auto& l : logits)
        l /= sum;
}

static bark_token gpt_multinomial_sample(
        std::vector<float> & logits,
        std::mt19937       & rng,
        float                temp,
        float              * eos_p) {
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

static bark_token gpt_argmax_sample(std::vector<float>& logits, float* eos_p) {
    int n_logits = logits.size();

    // testing purposes
    for (auto& l : logits)
        l /= 0.7f;

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
    std::mt19937       & rng,
    float                temp,
    float              * eos_p,
    int64_t            * t_sample_us,
    int64_t            * n_sample) {
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

static bool ggml_quantize_weights(
    std::ifstream                  & fin,
    std::ofstream                  & fout,
    const ggml_ftype                 ftype,
    const std::vector<std::string> & to_quant,
    const std::vector<std::string> & to_skip) {

    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_MOSTLY_Q2_K: qtype = GGML_TYPE_Q2_K; break;
        case GGML_FTYPE_MOSTLY_Q3_K: qtype = GGML_TYPE_Q3_K; break;
        case GGML_FTYPE_MOSTLY_Q4_K: qtype = GGML_TYPE_Q4_K; break;
        case GGML_FTYPE_MOSTLY_Q5_K: qtype = GGML_TYPE_Q5_K; break;
        case GGML_FTYPE_MOSTLY_Q6_K: qtype = GGML_TYPE_Q6_K; break;
        case GGML_FTYPE_UNKNOWN:
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
        case GGML_FTYPE_MOSTLY_IQ2_XXS:
        case GGML_FTYPE_MOSTLY_IQ2_XS:
        case GGML_FTYPE_MOSTLY_IQ2_S:
        case GGML_FTYPE_MOSTLY_IQ3_XXS:
        case GGML_FTYPE_MOSTLY_IQ3_S:
        case GGML_FTYPE_MOSTLY_IQ1_S:
        case GGML_FTYPE_MOSTLY_IQ4_NL:
        case GGML_FTYPE_MOSTLY_IQ4_XS:
        case GGML_FTYPE_MOSTLY_IQ1_M:
        case GGML_FTYPE_MOSTLY_BF16:
        case GGML_FTYPE_MOSTLY_Q4_0_4_4:
        case GGML_FTYPE_MOSTLY_Q4_0_4_8:
        case GGML_FTYPE_MOSTLY_Q4_0_8_8:
                {
                    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
                    return false;
                }
    };

    if (!ggml_is_quantized(qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ggml_type_name(qtype));
        return false;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    int32_t n_tensors = 0;
    read_safe(fin, n_tensors);
    write_safe(fout, n_tensors);

    for (int i = 0; i < n_tensors; i++) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        read_safe(fin, n_dims);
        read_safe(fin, length);
        read_safe(fin, ttype);

        int32_t nelements = 1;
        int32_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i) {
            read_safe(fin, ne[i]);
            nelements *= ne[i];
        }

        std::string name(length, 0);
        fin.read(&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type)ttype));

        bool quantize = false;

        // check if we should quantize this tensor
        for (const auto& s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto& s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_type_name((ggml_type)ttype));
                return false;
            }

            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                fin.read(reinterpret_cast<char*>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                fin.read(reinterpret_cast<char*>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
        } else {
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

            data_u8.resize(nelements * bpe);
            fin.read(reinterpret_cast<char*>(data_u8.data()), nelements * bpe);
        }

        write_safe(fout, n_dims);
        write_safe(fout, length);
        write_safe(fout, ttype);

        for (int i = 0; i < n_dims; ++i) {
            write_safe(fout, ne[i]);
        }

        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements);  // for quantization

            size_t cur_size = 0;

            switch ((ggml_type) ttype) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    {
                        cur_size = ggml_quantize_chunk((ggml_type) ttype, data_f32.data(), work.data(), 0, nelements/ne[0], ne[0], nullptr);
                    } break;
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I8:
                case GGML_TYPE_I16:
                case GGML_TYPE_I32:
                case GGML_TYPE_I64:
                case GGML_TYPE_F64:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_Q8_K:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_BF16:
                case GGML_TYPE_Q4_0_4_4:
                case GGML_TYPE_Q4_0_4_8:
                case GGML_TYPE_Q4_0_8_8:
                case GGML_TYPE_TQ1_0:
                case GGML_TYPE_TQ2_0:
                case GGML_TYPE_COUNT:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                        return false;
                    }
            }

            fout.write(reinterpret_cast<char*>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB\n", nelements * sizeof(float) / 1024.0 / 1024.0, cur_size / 1024.0 / 1024.0);
        } else {
            printf("size = %8.3f MB\n", data_u8.size() / 1024.0 / 1024.0);
            fout.write(reinterpret_cast<char*>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new / 1024.0 / 1024.0, ftype, ggml_type_name(qtype));

    return true;
}

static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string strip_accents(const std::string& in_str) {
    std::string out_str;
    std::map<std::string, char> accent_map = {
        {"À", 'A'},
        {"Á", 'A'},
        {"Â", 'A'},
        {"Ã", 'A'},
        {"Ä", 'A'},
        {"Å", 'A'},
        {"à", 'a'},
        {"á", 'a'},
        {"â", 'a'},
        {"ã", 'a'},
        {"ä", 'a'},
        {"å", 'a'},
        {"È", 'E'},
        {"É", 'E'},
        {"Ê", 'E'},
        {"Ë", 'E'},
        {"è", 'e'},
        {"é", 'e'},
        {"ê", 'e'},
        {"ë", 'e'},
        {"Ì", 'I'},
        {"Í", 'I'},
        {"Î", 'I'},
        {"Ï", 'I'},
        {"ì", 'i'},
        {"í", 'i'},
        {"î", 'i'},
        {"ï", 'i'},
        {"Ò", 'O'},
        {"Ó", 'O'},
        {"Ô", 'O'},
        {"Õ", 'O'},
        {"Ö", 'O'},
        {"ò", 'o'},
        {"ó", 'o'},
        {"ô", 'o'},
        {"õ", 'o'},
        {"ö", 'o'},
        {"Ù", 'U'},
        {"Ú", 'U'},
        {"Û", 'U'},
        {"Ü", 'U'},
        {"ù", 'u'},
        {"ú", 'u'},
        {"û", 'u'},
        {"ü", 'u'},
        {"Ý", 'Y'},
        {"ý", 'y'},
        {"Ç", 'C'},
        {"ç", 'c'},
        {"Ñ", 'N'},
        {"ñ", 'n'},
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
    const bark_vocab * vocab,
    const char       * text,
    int32_t          * tokens,
    int32_t          * n_tokens,
    int32_t            n_max_tokens) {
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
    for (const auto& word : words) {
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

static void bark_tokenize_input(struct bark_context* bctx, const std::string& text) {
    auto & model       = bctx->text_model.semantic_model;
    bark_vocab * vocab = &bctx->text_model.vocab;

    auto & params = bctx->params;

    int32_t block_size   = model.hparams.block_size;
    int32_t max_ctx_size = std::min(block_size, 256);
    int32_t n_tokens;

    bark_sequence tokens(max_ctx_size);
    bert_tokenize(vocab, text.data(), tokens.data(), &n_tokens, max_ctx_size);

    for (int i = 0; i < (int)tokens.size(); i++)
        tokens[i] += params.text_encoding_offset;

    if (n_tokens < max_ctx_size) {
        for (int i = n_tokens; i < max_ctx_size; i++)
            tokens[i] = params.text_pad_token;
    } else if (n_tokens > max_ctx_size) {
        fprintf(stderr, "%s: input sequence is too long (%d > 256), truncating sequence", __func__, n_tokens);
    }

    tokens.resize(max_ctx_size);

    // semantic history
    for (int i = 0; i < 256; i++)
        tokens.push_back(params.semantic_pad_token);
    tokens.push_back(params.semantic_infer_token);

    assert(tokens.size() == 256 + 256 + 1);

    bctx->tokens = tokens;

    printf("%s: prompt: '%s'\n", __func__, text.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, bctx->tokens.size());
    for (int i = 0; i < std::min(8, (int)bctx->tokens.size()); i++) {
        printf("%d ", bctx->tokens[i]);
    }
    printf("\n\n");
}

static bool bark_vocab_load(std::ifstream & fin, bark_vocab * vocab) {
    int32_t n_vocab;
    read_safe(fin, n_vocab);

    std::string word;
    std::vector<char> tmp;

    tmp.reserve(128);

    for (int i = 0; i < n_vocab; i++) {
        uint32_t len;
        read_safe(fin, len);

        if (len > 0) {
            tmp.resize(len);
            fin.read(&tmp[0], tmp.size());  // read to buffer
            word.assign(&tmp[0], tmp.size());
        } else {
            word = "";
        }

        vocab->token_to_id[word] = i;
        vocab->id_to_token[i]    = word;
    }

    return true;
}

static bool bark_model_load(std::ifstream & fin,
                            gpt_model     & model,
                            int             n_gpu_layers,
                            bark_verbosity_level verbosity) {
    // load hparams
    {
        auto & hparams = model.hparams;

        read_safe(fin, hparams.n_layer);
        read_safe(fin, hparams.n_head);
        read_safe(fin, hparams.n_embd);
        read_safe(fin, hparams.block_size);
        read_safe(fin, hparams.bias);
        read_safe(fin, hparams.n_in_vocab);
        read_safe(fin, hparams.n_out_vocab);
        read_safe(fin, hparams.n_lm_heads);
        read_safe(fin, hparams.n_wtes);
        read_safe(fin, hparams.ftype);

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
            printf("%s: n_in_vocab  = %d\n", __func__, hparams.n_in_vocab);
            printf("%s: n_out_vocab = %d\n", __func__, hparams.n_out_vocab);
            printf("%s: block_size  = %d\n", __func__, hparams.block_size);
            printf("%s: bias        = %d\n", __func__, hparams.bias);
            printf("%s: n_embd      = %d\n", __func__, hparams.n_embd);
            printf("%s: n_head      = %d\n", __func__, hparams.n_head);
            printf("%s: n_layer     = %d\n", __func__, hparams.n_layer);
            printf("%s: n_lm_heads  = %d\n", __func__, hparams.n_lm_heads);
            printf("%s: n_wtes      = %d\n", __func__, hparams.n_wtes);
            printf("%s: ftype       = %d\n", __func__, hparams.ftype);
            printf("%s: qntvr       = %d\n", __func__, qntvr);
        }

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file (bad ftype value %d)\n",
                __func__, model.hparams.ftype);
        return false;
    }

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
        const int bias        = hparams.bias;

        buffer_size += n_embd * ggml_type_size(GGML_TYPE_F32);  // ln_f_g

        buffer_size += n_wtes * n_in_vocab * n_embd * ggml_type_size(wtype);       // wtes
        buffer_size += block_size * n_embd * ggml_type_size(GGML_TYPE_F32);        // wpe
        buffer_size += n_lm_heads * n_out_vocab * n_embd * ggml_type_size(wtype);  // lm_head

        buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32));  // ln_1_g
        buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32));  // ln_2_g

        buffer_size += n_layer * (3 * n_embd * n_embd * ggml_type_size(wtype));  // c_attn_attn_w
        buffer_size += n_layer * (n_embd * n_embd * ggml_type_size(wtype));      // c_attn_proj_w

        buffer_size += n_layer * (4 * n_embd * n_embd * ggml_type_size(wtype));  // c_mlp_fc_w
        buffer_size += n_layer * (4 * n_embd * n_embd * ggml_type_size(wtype));  // c_mlp_proj_w

        if (bias) {
            buffer_size += n_embd * ggml_type_size(GGML_TYPE_F32);  // ln_f_b

            buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32));  // ln_1_b
            buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32));  // ln_2_b

            buffer_size += n_layer * (3 * n_embd * ggml_type_size(GGML_TYPE_F32));  // c_attn_attn_b
            buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32));      // c_attn_proj_b

            buffer_size += n_layer * (4 * n_embd * ggml_type_size(GGML_TYPE_F32));  // c_mlp_fc_b
            buffer_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32));      // c_mlp_proj_b
        }

        buffer_size += 10ull * MB;  // object overhead

        n_tensors = (1 +            // ln_f_g
                     n_wtes + 1 +   // wtes, wpe
                     2 * n_layer +  // ln_1_g, ln_2_g
                     2 * n_layer +  // c_attn_attn_w, c_attn_proj_w
                     2 * n_layer +  // c_mlp_fc_w, c_mlp_proj_w
                     n_lm_heads +   // lm_head
                     2              // memory_k, memory_v
        );

        if (bias) {
            n_tensors += 1;            // ln_f_b
            n_tensors += 2 * n_layer;  // ln_1_b, ln_2_b
            n_tensors += 4 * n_layer;  // c_attn_attn_b, c_attn_proj_b, c_mlp_fc_b, c_mlp_proj_b
        }

        if (verbosity == bark_verbosity_level::HIGH) {
            printf("%s: ggml tensor size = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
            printf("%s: ggml ctx size = %6.2f MB\n", __func__, buffer_size / (1024.0 * 1024.0));
        }
    }

    auto & ctx = model.ctx_w;

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!model.backend) {
        // fallback to CPU backend
        if (verbosity == bark_verbosity_level::HIGH) {
            fprintf(stderr, "%s: no backend specified, using CPU backend\n", __func__);
        }
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        if (verbosity == bark_verbosity_level::HIGH) {
            fprintf(stderr, "%s: failed to initialize CPU backend\n", __func__);
        }

        return false;
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
        const int bias        = hparams.bias;

        model.layers.resize(n_layer);
        model.lm_heads.resize(n_lm_heads);
        model.wtes.resize(n_wtes);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        if (bias)
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

        model.tensors["model/wpe"] = model.wpe;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 3 * n_embd);
            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.c_mlp_fc_w   = ggml_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd);
            layer.c_mlp_proj_w = ggml_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd);

            if (bias) {
                layer.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
                layer.ln_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

                layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * n_embd);
                layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

                layer.c_mlp_fc_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_embd);
                layer.c_mlp_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            }

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"] = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"] = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"] = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"] = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"] = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"] = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"] = layer.c_mlp_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"] = layer.c_mlp_proj_b;
        }
    }

    // allocate weights buffer
    model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);

    // key + value memory
    {
        auto & ctx = model.ctx_kv;

        // create the ggml context for key + value memory
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ctx = ggml_init(params);
            if (!ctx) {
                fprintf(stderr, "%s: ggml_init() failed\n", __func__);
                return false;
            }
        }

        const auto & hparams = model.hparams;

        const int n_embd      = hparams.n_embd;
        const int n_layer     = hparams.n_layer;
        const int block_size  = hparams.block_size;

        const int n_lm_heads  = hparams.n_lm_heads;
        const int n_wtes      = hparams.n_wtes;

        const int n_mem      = n_layer * block_size;
        const int n_elements = n_embd * n_mem;

        if (n_lm_heads == 1 && n_wtes == 1) {
            // hack: if one LM head and one token embedding layer, we are loading weights
            // of the text and coarse encoder. In this case, we need KV cache.
            // for fine encoder, no need for KV cache, skip this part.
            model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
            model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

            const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

            if (verbosity == bark_verbosity_level::HIGH) {
                printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);
            }

            // allocate the KV memory in a backend buffer
            model.buffer_kv = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
        }
    }

    // load weights
    {
        size_t total_size = 0;

        std::vector<char> read_buf;

        int32_t n_tensors;
        read_safe(fin, n_tensors);

        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
            printf("%s: loading %d tensors\n", __func__, n_tensors);
        }

        for (int i = 0; i < n_tensors; i++) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(fin, n_dims);
            read_safe(fin, length);
            read_safe(fin, ttype);

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
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

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            if (ggml_backend_buffer_is_host(model.buffer_w)) {
                // for the CPU and Metal backends, we can read directly into the device memory
                fin.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            if (verbosity == bark_verbosity_level::HIGH) {
                printf("%48s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], "float", ggml_nbytes(tensor) / 1024.0 / 1024.0);
            }

            total_size += ggml_nbytes(tensor);
        }

        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
            printf("%s: model size  = %8.2f MB\n", __func__, total_size / 1024.0 / 1024.0);
        }

        model.memsize = total_size;
    }

    return true;
}

static bool bark_load_model_from_file(
    const std::string    & fname,
    struct bark_context  * bctx,
    bark_verbosity_level   verbosity) {
    if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
        printf("%s: loading model from '%s'\n", __func__, fname.c_str());
    }

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char*)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // vocab
    {
        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
            printf("%s: reading bark vocab\n", __func__);
        }

        if (!bark_vocab_load(fin, &bctx->text_model.vocab)) {
            fprintf(stderr, "%s: failed to load vocab\n", __func__);
            return false;
        }
    }

    int n_gpu_layers = bctx->n_gpu_layers;

    // text
    {
        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
            printf("%s: reading bark text model\n", __func__);
        }

        if (!bark_model_load(fin, bctx->text_model.semantic_model, n_gpu_layers, verbosity)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad text)\n", __func__, fname.c_str());
            return false;
        }
    }

    // coarse
    {
        if (!bark_model_load(fin, bctx->text_model.coarse_model, n_gpu_layers, verbosity)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad coarse)\n", __func__, fname.c_str());
            return false;
        }
    }

    // fine
    {
        if (!bark_model_load(fin, bctx->text_model.fine_model, n_gpu_layers, verbosity)) {
            fprintf(stderr, "%s: invalid model file '%s' (bad fine)\n", __func__, fname.c_str());
            return false;
        }
    }

    // codec model
    {
        // not optimal: we close the file and reopen it using Encodec.cpp library with a
        // specific offset
        const int offset = fin.tellg();
        fin.close();

        bctx->encodec_ctx = encodec_load_model(fname.c_str(), offset, n_gpu_layers);
        if (!bctx->encodec_ctx) {
            fprintf(stderr, "%s: invalid model file '%s' (bad encodec)\n", __func__, fname.c_str());
            return false;
        }
    }

    printf("\n");

    return true;
}

struct bark_context* bark_load_model(const char* model_path, struct bark_context_params params, uint32_t seed) {
    ggml_time_init();
    int64_t t_load_start_us = ggml_time_us();

    struct bark_context * bctx = new bark_context();

    bctx->text_model = bark_model();

    std::string model_path_str(model_path);
    if (!bark_load_model_from_file(model_path_str, bctx, params.verbosity)) {
        fprintf(stderr, "%s: failed to load model weights from '%s'\n", __func__, model_path);
        return nullptr;
    }

    bctx->rng = std::mt19937(seed);
    bctx->params = params;
    bctx->stats.t_load_us = ggml_time_us() - t_load_start_us;

    return bctx;
}

static struct ggml_cgraph * bark_build_gpt_graph(
          const gpt_model * model,
          bark_sequence   & tokens,
          int             * n_past,
          bool              merge_ctx) {
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
    const int bias    = hparams.bias;

    static size_t buf_size = ggml_tensor_overhead() * BARK_MAX_NODES + ggml_graph_overhead_custom(BARK_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, BARK_MAX_NODES, false);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_set_input(input);
    ggml_set_name(input, "input");

    struct ggml_tensor * tok_emb;

    if (*n_past > 0) {
        assert(N == 1);
        tok_emb = ggml_get_rows(ctx0, model->wtes[0], input);
    } else {
        if (merge_ctx) {
            assert(N == 256 + 256 + 1);
            N -= 256;
        } else {
            assert(N <= n_ctx);
        }

        if (merge_ctx) {
            struct ggml_tensor * seq_embd = ggml_get_rows(ctx0, model->wtes[0], ggml_view_1d(ctx0, input, 256, 0));
            struct ggml_tensor * ctx_embd = ggml_get_rows(ctx0, model->wtes[0], ggml_view_1d(ctx0, input, 256, 256 * ggml_element_size(input)));
            struct ggml_tensor * rem_embd = ggml_get_rows(ctx0, model->wtes[0], ggml_view_1d(ctx0, input, 1, 512 * ggml_element_size(input)));

            struct ggml_tensor * cat_emb = ggml_add(ctx0, seq_embd, ctx_embd);

            tok_emb = ggml_new_tensor_2d(ctx0, cat_emb->type, cat_emb->ne[0], cat_emb->ne[1] + rem_embd->ne[1]);
            ggml_set_input(tok_emb);

            tok_emb = ggml_set_1d(ctx0, tok_emb, cat_emb, 0);
            tok_emb = ggml_set_1d(ctx0, tok_emb, rem_embd, cat_emb->ne[0] * cat_emb->ne[1] * ggml_element_size(cat_emb));
        } else {
            tok_emb = ggml_get_rows(ctx0, model->wtes[0], input);
        }
    }

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_set_input(position);
    ggml_set_name(position, "position");

    // wte + wpe
    struct ggml_tensor * inpL = ggml_add(ctx0, tok_emb, ggml_get_rows(ctx0, model->wpe, position));

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_norm(ctx0, inpL, EPS_NORM);

            // cur = ln_1_g*cur + ln_1_b
            cur = ggml_mul(ctx0, cur, model->layers[il].ln_1_g);

            if (bias) {
                cur = ggml_add(ctx0, cur, model->layers[il].ln_1_b);
            }
        }

        // attn
        {
            cur = ggml_mul_mat(ctx0,
                               model->layers[il].c_attn_attn_w,
                               cur);

            if (bias) {
                cur = ggml_add(ctx0, cur, model->layers[il].c_attn_attn_b);
            }
        }

        // self-attention
        {
            struct ggml_tensor* Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0 * sizeof(float) * n_embd);
            struct ggml_tensor* Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1 * sizeof(float) * n_embd);
            struct ggml_tensor* Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2 * sizeof(float) * n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor* k = ggml_view_1d(ctx0, model->memory_k, N * n_embd, (ggml_element_size(model->memory_k) * n_embd) * (il * n_ctx + *n_past));
                struct ggml_tensor* v = ggml_view_1d(ctx0, model->memory_v, N * n_embd, (ggml_element_size(model->memory_v) * n_embd) * (il * n_ctx + *n_past));

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor* Q =
                ggml_permute(ctx0,
                             ggml_cpy(ctx0,
                                      Qcur,
                                      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd / n_head, n_head, N)),
                             0, 2, 1, 3);

            struct ggml_tensor* K =
                ggml_permute(ctx0,
                             ggml_reshape_3d(ctx0,
                                             ggml_view_1d(ctx0, model->memory_k, (*n_past + N) * n_embd, il * n_ctx * ggml_element_size(model->memory_k) * n_embd),
                                             n_embd / n_head, n_head, *n_past + N),
                             0, 2, 1, 3);

            struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);

            struct ggml_tensor* KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f/sqrtf(float(n_embd)/n_head));

            struct ggml_tensor* KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, *n_past);

            struct ggml_tensor* KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            struct ggml_tensor* V_trans =
                ggml_cpy(ctx0,
                         ggml_permute(ctx0,
                                      ggml_reshape_3d(ctx0,
                                                      ggml_view_1d(ctx0, model->memory_v, (*n_past + N) * n_embd, il * n_ctx * ggml_element_size(model->memory_v) * n_embd),
                                                      n_embd / n_head, n_head, *n_past + N),
                                      1, 2, 0, 3),
                         ggml_new_tensor_3d(ctx0, model->memory_v->type, *n_past + N, n_embd / n_head, n_head));

            struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            struct ggml_tensor* KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0, model->layers[il].c_attn_proj_w, cur);

            if (bias) {
                cur = ggml_add(ctx0, cur, model->layers[il].c_attn_proj_b);
            }
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
                cur = ggml_mul(ctx0, cur, model->layers[il].ln_2_g);

                if (bias) {
                    cur = ggml_add(ctx0, cur, model->layers[il].ln_2_b);
                }
            }

            // cur = fc_w*cur + fc_b
            cur = ggml_mul_mat(ctx0, model->layers[il].c_mlp_fc_w, cur);

            if (bias) {
                cur = ggml_add(ctx0, cur, model->layers[il].c_mlp_fc_b);
            }

            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0, model->layers[il].c_mlp_proj_w, cur);

            if (bias) {
                cur = ggml_add(ctx0, cur, model->layers[il].c_mlp_proj_b);
            }
        }

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL, EPS_NORM);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_mul(ctx0, inpL, model->ln_f_g);

        if (bias) {
            inpL = ggml_add(ctx0, inpL, model->ln_f_b);
        }
    }

    inpL = ggml_mul_mat(ctx0,
                        model->lm_heads[0],
                        ggml_view_1d(ctx0, inpL, inpL->ne[0], (inpL->ne[1] - 1) * inpL->nb[1]));
    ggml_set_name(inpL, "logits");
    ggml_set_output(inpL);

    ggml_build_forward_expand(gf, inpL);

    ggml_free(ctx0);

    return gf;
}

static ggml_cgraph * bark_build_fine_gpt_graph(
    const gpt_model * model,
    bark_sequence   & tokens,
    int               codebook_idx,
    int               n_fine_codebooks) {
    // tokens: [n_channels, N]
    const int N = tokens.size() / n_fine_codebooks;
    const int n_channels = n_fine_codebooks;

    const auto & hparams = model->hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.block_size;
    const int n_head  = hparams.n_head;

    const int n_codes_given = hparams.n_codes_given;

    assert(N <= n_ctx);
    assert(codebook_idx > 0);

    static size_t buf_size = ggml_tensor_overhead() * BARK_MAX_NODES + ggml_graph_overhead_custom(BARK_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, BARK_MAX_NODES, false);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, N, n_channels);
    ggml_set_input(input);
    ggml_set_name(input, "input");

    struct ggml_tensor * tok_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N);
    ggml_set_name(tok_emb, "tok_emb");

    for (int wte_ix = 0; wte_ix < codebook_idx + 1; wte_ix++) {
        struct ggml_tensor * cur = ggml_get_rows(ctx0,
                                                 model->wtes[wte_ix],
                                                 ggml_view_1d(ctx0, input, N, wte_ix * input->nb[1]));

        tok_emb = ggml_add(ctx0, tok_emb, cur);
    }

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_set_input(position);
    ggml_set_name(position, "position");

    struct ggml_tensor * pos_emb = ggml_get_rows(ctx0, model->wpe, position);

    // wte + wpe
    struct ggml_tensor * inpL = ggml_add(ctx0, tok_emb, pos_emb);

    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_norm(ctx0, inpL, EPS_NORM);

            // cur = ln_1_g*cur + ln_1_b
            cur = ggml_mul(ctx0, cur, model->layers[il].ln_1_g);
            cur = ggml_add(ctx0, cur, model->layers[il].ln_1_b);
        }

        // self-attention
        {
            // cur = attn_w*cur
            cur = ggml_mul_mat(ctx0, model->layers[il].c_attn_attn_w, cur);

            struct ggml_tensor* Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0 * sizeof(float) * n_embd);
            struct ggml_tensor* Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1 * sizeof(float) * n_embd);
            struct ggml_tensor* Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2 * sizeof(float) * n_embd);

            struct ggml_tensor* Q =
                ggml_permute(ctx0,
                             ggml_cpy(ctx0,
                                      Qcur,
                                      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd / n_head, n_head, N)),
                             0, 2, 1, 3);

            struct ggml_tensor* K =
                ggml_permute(ctx0,
                             ggml_cpy(ctx0,
                                      Kcur,
                                      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd / n_head, n_head, N)),
                             0, 2, 1, 3);

            struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);

            struct ggml_tensor* KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f/sqrtf(float(n_embd)/n_head));

            struct ggml_tensor* KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_scaled);

            struct ggml_tensor* V_trans =
                ggml_cont(ctx0,
                          ggml_permute(ctx0,
                                       ggml_cpy(ctx0,
                                                Vcur,
                                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd / n_head, n_head, N)),
                                       1, 2, 0, 3));

            struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            struct ggml_tensor* KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // [n_embd, N]
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // cur = proj_w*cur
            cur = ggml_mul_mat(ctx0, model->layers[il].c_attn_proj_w, cur);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor* inpFF = cur;

        // feed-forward
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, EPS_NORM);

                cur = ggml_mul(ctx0, cur, model->layers[il].ln_2_g);
                cur = ggml_add(ctx0, cur, model->layers[il].ln_2_b);
            }

            // cur = fc_w*cur
            cur = ggml_mul_mat(ctx0, model->layers[il].c_mlp_fc_w, cur);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // cur = proj_w*cur
            cur = ggml_mul_mat(ctx0, model->layers[il].c_mlp_proj_w, cur);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL, EPS_NORM);

        inpL = ggml_mul(ctx0, inpL, model->ln_f_g);
        inpL = ggml_add(ctx0, inpL, model->ln_f_b);
    }

    // inpL = WTE * inpL
    struct ggml_tensor* lm_head = model->lm_heads[codebook_idx - n_codes_given];
    inpL = ggml_mul_mat(ctx0, lm_head, inpL);

    ggml_set_output(inpL);
    ggml_set_name(inpL, "logits");

    ggml_build_forward_expand(gf, inpL);

    ggml_free(ctx0);

    return gf;
}

static bool bark_eval_encoder_internal(
    gpt_model          & model,
    ggml_gallocr_t       allocr,
    bark_sequence      & input_sequence,
    std::vector<float> & logits,
    int                * n_past,
    bool                 merge_ctx,
    int                  n_threads) {
    auto & hparams = model.hparams;
    const int n_vocab = hparams.n_out_vocab;
    int N = input_sequence.size();

    const int64_t t_predict_us_start = ggml_time_us();

    struct ggml_cgraph * gf = bark_build_gpt_graph(&model, input_sequence, n_past, merge_ctx);

    // allocate the graph tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set the graph inputs
    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(input, input_sequence.data(), 0, N * ggml_element_size(input));

    if (*n_past == 0 && merge_ctx) {
        // if we are merging contexts, we need to subtract the context tokens from the input size
        // this adapts the number of positions
        N -= 256;
    }

    struct ggml_tensor * position = ggml_graph_get_tensor(gf, "position");

    for (int i = 0; i < N; i++) {
        int32_t pos = i + *n_past;
        ggml_backend_tensor_set(position, &pos, i * sizeof(int32_t), sizeof(int32_t));
    }

    // set backend options
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    // run the computation
    ggml_backend_graph_compute(model.backend, gf);

    struct ggml_tensor * inpL = ggml_graph_get_tensor(gf, "logits");

    logits.resize(n_vocab);
    ggml_backend_tensor_get(inpL, logits.data(), 0, sizeof(float) * n_vocab);

    // updating n_past with N (-256 if merge_ctx)
    if (n_past) {
        *n_past += N;
    }

    model.t_predict_us += ggml_time_us() - t_predict_us_start;

    return true;
}

static bool bark_eval_text_encoder(struct bark_context * bctx, int n_threads) {
    bark_sequence input = bctx->tokens;
    bark_sequence output;

    auto & params = bctx->params;

    int32_t n_steps_text_encoder = params.n_steps_text_encoder;
    int32_t semantic_vocab_size  = params.semantic_vocab_size;
    int32_t semantic_pad_token   = params.semantic_pad_token;

    auto & model   = bctx->text_model.semantic_model;
    auto & allocr  = bctx->allocr;
    auto & hparams = model.hparams;

    const int n_vocab = hparams.n_out_vocab;
    float min_eos_p   = bctx->params.min_eos_p;
    float temp        = bctx->params.temp;

    std::vector<float> logits;
    logits.resize(n_vocab);

    float eos_p = 0;
    int n_past  = 0;

    for (int i = 0; i < n_steps_text_encoder; i++) {
        if (params.progress_callback) {
            const int progress_cur = 100 * (i + 1) / n_steps_text_encoder;

            params.progress_callback(
                bctx, bark_encoding_step::SEMANTIC, progress_cur, params.progress_callback_user_data);
        }

        if (!bark_eval_encoder_internal(model, allocr, input, logits, &n_past, true, n_threads)) {
            fprintf(stderr, "%s: Could not generate token\n", __func__);
            return false;
        }

        std::vector<float> relevant_logits(logits.begin(), logits.begin() + semantic_vocab_size);
        relevant_logits.push_back(logits[semantic_pad_token]);

        input.clear();

        bark_token next = gpt_sample(
            logits, bctx->rng, temp, &eos_p, &model.t_sample_us, &model.n_sample);

        if (next == semantic_vocab_size || eos_p >= min_eos_p)
            break;

        input.push_back(next);
        output.push_back(next);
    }

    bctx->semantic_tokens         = output;
    bctx->stats.n_sample_semantic = model.n_sample;

    return true;
}

bool bark_forward_text_encoder(struct bark_context * bctx, int n_threads) {
    const int64_t t_main_start_us = ggml_time_us();

    auto & model     = bctx->text_model.semantic_model;
    auto & allocr    = bctx->allocr;
    auto & hparams   = model.hparams;
    auto & verbosity = bctx->params.verbosity;

    // allocate the compute buffer
    {
        bctx->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst-case graph for memory usage estimation
        int n_past = 0;
        std::vector<bark_vocab::id> decoy_tokens(256 + 256 + 1, 0);
        struct ggml_cgraph * gf = bark_build_gpt_graph(&model, decoy_tokens, &n_past, true /* merge_ctx */);

        // pre-allocate the compute buffer for the worst case
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size =  ggml_gallocr_get_buffer_size(allocr, 0);

        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
            fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size / 1024.0 / 1024.0);
        }
    }

    if (!bark_eval_text_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward text encoder\n", __func__);
        return false;
    }

    model.t_main_us = ggml_time_us() - t_main_start_us;
    bctx->stats.t_semantic_us = model.t_main_us;

    bark_print_statistics(&model);

    ggml_gallocr_free(bctx->allocr);
    ggml_backend_buffer_free(bctx->buf_compute);

    return true;
}

static bool bark_eval_coarse_encoder(struct bark_context * bctx, int n_threads) {
    bark_codes out_coarse;
    bark_sequence out;

    bark_sequence input = bctx->semantic_tokens;

    auto & model   = bctx->text_model.coarse_model;
    auto & allocr  = bctx->allocr;
    auto & hparams = model.hparams;
    auto & params  = bctx->params;

    const int n_vocab = hparams.n_out_vocab;

    std::vector<float> logits;
    logits.resize(n_vocab);

    int max_coarse_history  = params.max_coarse_history;
    int sliding_window_size = params.sliding_window_size;
    int n_coarse_codebooks  = params.n_coarse_codebooks;
    int semantic_vocab_size = params.semantic_vocab_size;
    int codebook_size       = params.codebook_size;

    float coarse_rate_hz   = params.coarse_rate_hz;
    float semantic_rate_hz = params.semantic_rate_hz;

    int32_t coarse_semantic_pad_token = params.coarse_semantic_pad_token;
    int32_t coarse_infer_token        = params.coarse_infer_token;

    float temp = params.temp;

    float stc_ratio = coarse_rate_hz / semantic_rate_hz * n_coarse_codebooks;

    int max_semantic_history = floorf(max_coarse_history / stc_ratio);

    int n_steps = floorf(input.size() * stc_ratio / n_coarse_codebooks) * n_coarse_codebooks;
    assert(n_steps > 0);
    assert(n_steps % n_coarse_codebooks == 0);

    int n_window_steps = ceilf(static_cast<float>(n_steps) / sliding_window_size);

    int step_idx = 0;

    for (int i = 0; i < n_window_steps; i++) {
        int semantic_idx = roundf(step_idx / stc_ratio);

        bark_sequence input_in(
            input.begin() + std::max(semantic_idx - max_semantic_history, 0),
            input.end());

        size_t original_size = input_in.size();
        input_in.resize(256);

        // padding from the right side
        for (int ix = original_size; ix < 256; ix++) {
            input_in[ix] = coarse_semantic_pad_token;
        }
        input_in.push_back(coarse_infer_token);

        // concatenate input_in and input_coarse
        input_in.insert(
            input_in.end(),
            std::make_move_iterator(out.end() - std::min(max_coarse_history, (int)out.size())),
            std::make_move_iterator(out.end()));

        int n_past = 0;

        for (int j = 0; j < sliding_window_size; j++) {
            if (step_idx >= n_steps)
                continue;

            if (params.progress_callback) {
                const int progress_cur = 100 * (step_idx + 1) / n_steps;

                params.progress_callback(
                    bctx, bark_encoding_step::COARSE, progress_cur, params.progress_callback_user_data);
            }

            if (!bark_eval_encoder_internal(model, allocr, input_in, logits, &n_past, false, n_threads)) {
                fprintf(stderr, "%s: Could not generate token\n", __func__);
                return false;
            }

            input_in.clear();

            bool is_major = step_idx % n_coarse_codebooks == 0;
            int start_idx = semantic_vocab_size + (1 - is_major) * codebook_size;
            int end_idx   = semantic_vocab_size + (2 - is_major) * codebook_size;

            std::vector<float> relevant_logits(logits.begin() + start_idx, logits.begin() + end_idx);

            bark_token next = gpt_sample(
                relevant_logits, bctx->rng, temp, NULL, &model.t_sample_us, &model.n_sample);

            next += start_idx;

            input_in.push_back(next);
            out.push_back(next);

            step_idx += 1;
        }
    }

    assert((int)out.size() == n_steps);
    assert(out.size() % n_coarse_codebooks == 0);

    // out_coarse: [seq_length, n_codes]
    for (int i = 0; i < (int)out.size(); i += n_coarse_codebooks) {
        // this assumes N_COARSE_CODEBOOKS = 2
        bark_sequence _tmp = {
            out[i] - semantic_vocab_size,
            out[i + 1] - semantic_vocab_size - codebook_size};
        out_coarse.push_back(_tmp);
    }

    bctx->coarse_tokens         = out_coarse;
    bctx->stats.n_sample_coarse = model.n_sample;

    return true;
}

bool bark_forward_coarse_encoder(struct bark_context * bctx, int n_threads) {
    const int64_t t_main_start_us = ggml_time_us();

    auto & model     = bctx->text_model.coarse_model;
    auto & allocr    = bctx->allocr;
    auto & hparams   = model.hparams;
    auto & verbosity = bctx->params.verbosity;

    // allocate the compute buffer
    {
        // create a graph allocator with the backend's default buffer type
        bctx->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst-case graph for memory usage estimation
        int n_past = 0;
        std::vector<bark_vocab::id> decoy_tokens(hparams.block_size, 0);
        struct ggml_cgraph * gf = bark_build_gpt_graph(&model, decoy_tokens, &n_past, false /* merge_ctx */);

        // pre-allocate the compute buffer for the worst case (optional)
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH)
            fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size / 1024.0 / 1024.0);
    }

    if (!bark_eval_coarse_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward coarse encoder\n", __func__);
        return false;
    }

    model.t_main_us = ggml_time_us() - t_main_start_us;
    bctx->stats.t_coarse_us = model.t_main_us;

    bark_print_statistics(&model);

    ggml_backend_buffer_free(bctx->buf_compute);
    ggml_gallocr_free(bctx->allocr);

    return true;
}

static bool bark_eval_fine_encoder_internal(
    struct bark_context * bctx,
    bark_sequence       & input_sequence,
    std::vector<float>  & logits,
    int                   nn,
    int                   n_threads) {
    auto & model   = bctx->text_model.fine_model;
    auto & allocr  = bctx->allocr;
    auto & hparams = model.hparams;
    auto & params  = bctx->params;

    const int n_vocab    = hparams.n_out_vocab;
    const int block_size = hparams.block_size;

    const int n_fine_codebooks = params.n_fine_codebooks;

    const int N = input_sequence.size() / n_fine_codebooks;

    const int64_t t_predict_us_start = ggml_time_us();

    struct ggml_cgraph * gf = bark_build_fine_gpt_graph(&model, input_sequence, nn, n_fine_codebooks);

    // allocate the graph tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set the graph inputs
    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(input, input_sequence.data(), 0, N * n_fine_codebooks * ggml_element_size(input));

    struct ggml_tensor * tok_emb = ggml_graph_get_tensor(gf, "tok_emb");
    ggml_set_zero(tok_emb);

    struct ggml_tensor * position = ggml_graph_get_tensor(gf, "position");
    for (int i = 0; i < N; i++) {
        ggml_backend_tensor_set(position, &i, i * sizeof(int32_t), sizeof(int32_t));
    }

    // set backend options
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    // run the computation
    ggml_backend_graph_compute(model.backend, gf);

    struct ggml_tensor* inpL = ggml_graph_get_tensor(gf, "logits");

    ggml_backend_tensor_get(inpL, logits.data(), 0, sizeof(float) * n_vocab * block_size);

    model.t_predict_us += ggml_time_us() - t_predict_us_start;

    return true;
}

static bool bark_eval_fine_encoder(struct bark_context * bctx, int n_threads) {
    // input shape: [N, n_codes]
    bark_codes input = bctx->coarse_tokens;

    std::vector<float> logits;
    logits.resize(1024 * 1056);

    auto & model   = bctx->text_model.fine_model;
    auto & hparams = model.hparams;
    auto & params  = bctx->params;

    float temp = params.fine_temp;

    int32_t n_coarse_codebooks = params.n_coarse_codebooks;
    int32_t n_fine_codebooks   = params.n_fine_codebooks;
    int32_t codebook_size      = params.codebook_size;

    int n_coarse          = input[0].size();
    int original_seq_len  = input.size();
    int n_remove_from_end = 0;

    // channel padding
    for (int i = 0; i < (int)input.size(); i++) {
        for (int j = n_coarse_codebooks; j < n_fine_codebooks; j++) {
            input[i].push_back(codebook_size);
        }
    }

    // spatial padding if sequence is too short
    if (original_seq_len < 1024) {
        n_remove_from_end = 1024 - original_seq_len;
        for (int i = original_seq_len; i < 1024; i++) {
            bark_sequence _tmp(n_fine_codebooks, codebook_size);
            input.push_back(_tmp);
        }
    }

    int n_loops = std::max(0, (int)ceilf((input.size() - 1024) / 512.f)) + 1;

    bark_codes in_arr = input;  // [seq_length, n_codes]

    for (int n = 0; n < n_loops; n++) {
        int start_idx = std::min(n * 512, (int)in_arr.size() - 1024);
        int start_fill_idx = std::min(n * 512, (int)in_arr.size() - 512);
        int rel_start_fill_idx = start_fill_idx - start_idx;

        // in_buffer: [n_codes*seq_length] (sequences are contiguous)
        bark_sequence in_buffer;
        for (int i = 0; i < n_fine_codebooks; i++) {
            for (int j = start_idx; j < start_idx + 1024; j++) {
                in_buffer.push_back(in_arr[j][i]);
            }
        }

        for (int nn = n_coarse; nn < n_fine_codebooks; nn++) {
            if (params.progress_callback) {
                const int progress_cur = 100*(n*(n_fine_codebooks-n_coarse)+(nn-n_coarse+1))/(n_loops*(n_fine_codebooks-n_coarse));

                params.progress_callback(
                    bctx, bark_encoding_step::FINE, progress_cur, params.progress_callback_user_data);
            }

            if (!bark_eval_fine_encoder_internal(bctx, in_buffer, logits, nn, n_threads)) {
                fprintf(stderr, "%s: Could not generate token\n", __func__);
                return false;
            }
            for (int i = 0; i < 1024; i++) {
                std::vector<float> relevant_logits(
                    logits.begin() + i * 1056,
                    logits.begin() + (i + 1) * 1056);
                relevant_logits.resize(codebook_size);

                bark_token next = gpt_sample(
                    relevant_logits, bctx->rng, temp, NULL, &model.t_sample_us,
                    &model.n_sample);

                in_buffer[nn * 1024 + rel_start_fill_idx + i] = next;
            }
        }

        // transfer over info into model_in
        for (int nn = n_coarse; nn < n_fine_codebooks; nn++) {
            for (int j = 0; j < codebook_size - rel_start_fill_idx; j++) {
                in_arr[start_fill_idx + j][nn] = in_buffer[nn * 1024 + rel_start_fill_idx + j];
            }
        }
    }

    if (n_remove_from_end > 0) {
        in_arr.resize(in_arr.size() - n_remove_from_end);
    }

    assert(bctx->coarse_tokens.size() == in_arr.size());

    bctx->fine_tokens = in_arr;
    bctx->stats.n_sample_fine = model.n_sample;

    return true;
}

bool bark_forward_fine_encoder(struct bark_context * bctx, int n_threads) {
    const int64_t t_main_start_us = ggml_time_us();

    auto & model     = bctx->text_model.fine_model;
    auto & allocr    = bctx->allocr;
    auto & hparams   = model.hparams;
    auto & params    = bctx->params;
    auto & verbosity = bctx->params.verbosity;

    int32_t n_fine_codebooks = params.n_fine_codebooks;

    // allocate the compute buffer
    {
        // alignment required by the backend
        bctx->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst-case graph for memory usage estimation
        std::vector<bark_vocab::id> decoy_tokens(hparams.block_size * n_fine_codebooks, 0);
        struct ggml_cgraph* gf = bark_build_fine_gpt_graph(&model, decoy_tokens, 2 /* codebook_idx */, n_fine_codebooks);

        // pre-allocate the compute buffer for the worst case (optional)
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size =  ggml_gallocr_get_buffer_size(allocr, 0);

        if (verbosity == bark_verbosity_level::MEDIUM || verbosity == bark_verbosity_level::HIGH) {
            fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size / 1024.0 / 1024.0);
        }
    }

    if (!bark_eval_fine_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward coarse encoder\n", __func__);
        return false;
    }

    model.t_main_us = ggml_time_us() - t_main_start_us;
    bctx->stats.t_fine_us = model.t_main_us;

    bark_print_statistics(&model);

    ggml_backend_buffer_free(bctx->buf_compute);
    ggml_gallocr_free(bctx->allocr);

    return true;
}

static bool bark_forward_eval(struct bark_context * bctx, int n_threads) {
    if (!bark_forward_text_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward text encoder\n", __func__);
        return false;
    }

    if (!bark_forward_coarse_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward coarse encoder\n", __func__);
        return false;
    }

    if (!bark_forward_fine_encoder(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward fine encoder\n", __func__);
        return false;
    }

    return true;
}

bool bark_generate_audio(struct bark_context * bctx, const char * text, int n_threads) {
    if (!bctx) {
        fprintf(stderr, "%s: invalid bark context\n", __func__);
        return false;
    }

    bark_reset_statistics(bctx);

    int64_t t_start_eval_us = ggml_time_us();

    std::string text_str(text);
    bark_tokenize_input(bctx, text_str);

    if (!bark_forward_eval(bctx, n_threads)) {
        fprintf(stderr, "%s: failed to forward eval\n", __func__);
        return false;
    }

    auto & params = bctx->params;

    int32_t target_bandwidth = params.target_bandwidth;
    int32_t sample_rate = params.sample_rate;

    encodec_set_target_bandwidth(bctx->encodec_ctx, target_bandwidth);
    encodec_set_sample_rate(bctx->encodec_ctx, sample_rate);

    // current shape fine_tokens: [seq_length][n_channels], n_channels are contiguous
    // encodec expects shape fine_tokens: [n_channels][seq_length], time steps are contiguous
    std::vector<bark_vocab::id> encodec_tokens;

    for (int i = 0; i < (int)bctx->fine_tokens[0].size(); i++) {
        for (int j = 0; j < (int)bctx->fine_tokens.size(); j++) {
            encodec_tokens.push_back(bctx->fine_tokens[j][i]);
        }
    }

    if (!encodec_decompress_audio(bctx->encodec_ctx, encodec_tokens.data(), encodec_tokens.size(), n_threads)) {
        printf("%s: Could not generate waveform from tokens with Encodec\n", __func__);
        return false;
    }

    bctx->generated_audio     = encodec_get_audio(bctx->encodec_ctx);
    bctx->n_generated_samples = encodec_get_audio_size(bctx->encodec_ctx);

    bctx->stats.t_eval_us = ggml_time_us() - t_start_eval_us;

    return true;
}

static void bark_free_model(struct gpt_model * model) {
    if (!model)
        return;

    if (model->ctx_w)
        ggml_free(model->ctx_w);

    if (model->ctx_kv)
        ggml_free(model->ctx_kv);

    ggml_backend_buffer_free(model->buffer_w);
    ggml_backend_buffer_free(model->buffer_kv);
    ggml_backend_free(model->backend);
}

void bark_free(struct bark_context * bctx) {
    if (!bctx)
        return;

    encodec_free(bctx->encodec_ctx);

    bark_free_model(&bctx->text_model.semantic_model);
    bark_free_model(&bctx->text_model.coarse_model);
    bark_free_model(&bctx->text_model.fine_model);

    delete bctx;
}

struct bark_context_params bark_context_default_params() {
    struct bark_context_params result = {
        /*.verbosity                   =*/ bark_verbosity_level::LOW,
        /*.temp                        =*/ 0.7,
        /*.fine_temp                   =*/ 0.5,
        /*.min_eos_p                   =*/ 0.2,
        /*.sliding_window_size         =*/ 60,
        /*.max_coarse_history          =*/ 630,
        /*.sample_rate                 =*/ 24000,
        /*.target_bandwidth            =*/ 6,
        /*.cls_token_id                =*/ 101,
        /*.sep_token_id                =*/ 102,
        /*.n_steps_text_encoder        =*/ 768,
        /*.text_pad_token              =*/ 129595,
        /*.text_encoding_offset        =*/ 10048,
        /*.semantic_rate_hz            =*/ 49.9f,
        /*.semantic_pad_token          =*/ 10000,
        /*.semantic_vocab_size         =*/ 10000,
        /*.semantic_infer_token        =*/ 129599,
        /*.coarse_rate_hz              =*/ 75.0f,
        /*.coarse_infer_token          =*/ 12050,
        /*.coarse_semantic_pad_token   =*/ 12048,
        /*.n_coarse_codebooks          =*/ 2,
        /*.n_fine_codebooks            =*/ 8,
        /*.codebook_size               =*/ 1024,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
    };

    return result;
}

bool bark_model_weights_quantize(std::ifstream & fin, std::ofstream & fout, ggml_ftype ftype) {
    gpt_model model;
    gpt_hparams hparams;

    // load hparams
    {
        auto & hparams = model.hparams;

        read_safe(fin, hparams.n_layer);
        read_safe(fin, hparams.n_head);
        read_safe(fin, hparams.n_embd);
        read_safe(fin, hparams.block_size);
        read_safe(fin, hparams.bias);
        read_safe(fin, hparams.n_in_vocab);
        read_safe(fin, hparams.n_out_vocab);
        read_safe(fin, hparams.n_lm_heads);
        read_safe(fin, hparams.n_wtes);
        read_safe(fin, hparams.ftype);

        const int32_t qntvr_src = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        printf("%s: n_in_vocab  = %d\n", __func__, hparams.n_in_vocab);
        printf("%s: n_out_vocab = %d\n", __func__, hparams.n_out_vocab);
        printf("%s: block_size  = %d\n", __func__, hparams.block_size);
        printf("%s: bias        = %d\n", __func__, hparams.bias);
        printf("%s: n_embd      = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head      = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer     = %d\n", __func__, hparams.n_layer);
        printf("%s: n_lm_heads  = %d\n", __func__, hparams.n_lm_heads);
        printf("%s: n_wtes      = %d\n", __func__, hparams.n_wtes);
        printf("%s: ftype (src) = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr (src) = %d\n", __func__, qntvr_src);
        printf("%s: ftype (dst) = %d\n", __func__, ftype_dst);
        printf("%s: qntvr (dst) = %d\n", __func__, GGML_QNT_VERSION);

        write_safe(fout, hparams.n_layer);
        write_safe(fout, hparams.n_head);
        write_safe(fout, hparams.n_embd);
        write_safe(fout, hparams.block_size);
        write_safe(fout, hparams.bias);
        write_safe(fout, hparams.n_in_vocab);
        write_safe(fout, hparams.n_out_vocab);
        write_safe(fout, hparams.n_lm_heads);
        write_safe(fout, hparams.n_wtes);
        write_safe(fout, ftype_dst);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> to_quant = {
        "model/wte/.*",
        "model/lm_head/.*",
        "model/h.*/attn/c_attn/w",
        "model/h.*/attn/c_proj/w",
        "model/h.*/mlp/c_fc/w",
        "model/h.*/mlp/c_proj/w",
    };

    if (!ggml_quantize_weights(fin, fout, ftype, to_quant, {})) {
        fprintf(stderr, "%s: failed to quantize model\n", __func__);
        return false;
    }

    return true;
}

bool bark_model_quantize(const char * fname_inp, const char * fname_out, enum ggml_ftype ftype) {
    printf("%s: loading model from '%s'\n", __func__, fname_inp);

    std::string fname_inp_str(fname_inp);
    std::string fname_out_str(fname_out);

    auto fin = std::ifstream(fname_inp_str, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp);
        return false;
    }

    auto fout = std::ofstream(fname_out_str, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out);
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char*)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp);
            return false;
        }

        fout.write((char*)&magic, sizeof(magic));
    }

    // transfer vocab data from fin to fout
    {
        uint32_t n_vocab;
        read_safe(fin, n_vocab);
        write_safe(fout, n_vocab);

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(fin, len);
            write_safe(fout, len);

            word.resize(len);
            fin.read((char*)word.data(), len);
            fout.write((char*)word.data(), len);
        }
    }

    // text model
    if (!bark_model_weights_quantize(fin, fout, ftype)) {
        fprintf(stderr, "%s: failed to quantize text model\n", __func__);
        return false;
    }

    // coarse model
    if (!bark_model_weights_quantize(fin, fout, ftype)) {
        fprintf(stderr, "%s: failed to quantize coarse model\n", __func__);
        return false;
    }

    // fine model
    if (!bark_model_weights_quantize(fin, fout, ftype)) {
        fprintf(stderr, "%s: failed to quantize fine model\n", __func__);
        return false;
    }

    // neural codec (not quantized, since this seriously degrates the audio quality)
    // copy the rest of fin to fout
    char c;
    while (fin.get(c)) {
        fout.put(c);
    }

    fin.close();
    fout.close();

    return true;
}

float* bark_get_audio_data(struct bark_context * bctx) {
    if (!bctx)
        return nullptr;
    return bctx->generated_audio;
}

int bark_get_audio_data_size(struct bark_context * bctx) {
    if (!bctx || bctx->generated_audio == NULL)
        return 0;
    return bctx->n_generated_samples;
}

int64_t bark_get_load_time(struct bark_context * bctx) {
    if (!bctx)
        return 0;
    return bctx->stats.t_load_us;
}

int64_t bark_get_eval_time(struct bark_context * bctx) {
    if (!bctx)
        return 0;
    return bctx->stats.t_eval_us;
}

void bark_reset_statistics(struct bark_context * bctx) {
    if (!bctx)
        return;
    memset(&bctx->stats, 0, sizeof(bark_statistics));
}
