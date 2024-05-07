"""This script converts a Bark model to CoreML format.

Note that the conversion is not trivial. There are slight discrepenacies between
the forward pass of a PyTorch model and a CoreML model. Furthermore, to fully utilize
the ANE, we need to apply some tensor reshaping. These optimizations are fully documented
here:
    https://machinelearning.apple.com/research/neural-engine-transformers
"""
import argparse
import json
from typing import Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
from bark.model import CausalSelfAttention, MLP, Block, GPT
from bark.model_fine import NonCausalSelfAttention, FineGPT
from ane_transformers.reference.layer_norm import LayerNormANE as LayerNormANEBase
import coremltools as ct
from coremltools.models.neural_network.quantization_utils import quantize_weights


EPS = 1e-5

def rename_keys(hparams):
    """Rename keys to match the expected keys in the bark PyPi package."""
    hparams["n_embd"] = hparams.pop("hidden_size")
    hparams["n_head"] = hparams.pop("num_heads")
    hparams["n_layer"] = hparams.pop("num_layers")
    return hparams


def rename_checkpoint_key(checkpoint_key):
    return checkpoint_key \
                .replace("fine_acoustics.", "") \
                .replace("input_embeds_layers.", "transformer.wtes.") \
                .replace("position_embeds_layer", "transformer.wpe") \
                .replace("layers.", "transformer.blocks.") \
                .replace("layernorm", "ln") \
                .replace("attn.att_proj", "attn.c_attn") \
                .replace("attn.out_proj", "attn.c_proj") \
                .replace("mlp.in_proj", "mlp.c_fc") \
                .replace("mlp.out_proj", "mlp.c_proj") \
                .replace("ln_final", "transformer.ln_f")

# The native torch.nn.Transformer and many other PyTorch implementations use either the
# (B, S, C) or the (S, B, C) data formats, which are both channels-last and 3D data
# formats. These data formats are compatible with nn.Linear layers, which constitute a
# major chunk of compute in the Transformer. To migrate to the desirable (B, C, 1, S)
# data format, we swap all nn.Linear layers with nn.Conv2d layers.
# This function adapts the weights of nn.Linear layers to nn.Conv2d layers.
def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights"""
    for k in state_dict:
        is_head = all(substr in k for substr in ['lm_heads', '.weight'])
        is_attention = all(substr in k for substr in ['attn', '.weight'])
        is_mlp = all(substr in k for substr in ['mlp', '.weight'])

        if (is_attention or is_mlp or is_head) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]


# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
# Source: https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/huggingface/distilbert.py#L25
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
    return state_dict


class AttrDict:
    def __init__(self, data):
        self.__dict__.update(data)

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self.__dict__[key] = value


class LayerNormANE(LayerNormANEBase):

    def __init__(self, num_channels, use_bias, eps):
        super().__init__(num_channels, eps=eps)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)

        if use_bias:
            self.bias = torch.zeros(num_channels)


class CausalSelfAttentionANE(CausalSelfAttention):

    def __init__(self, config):
        super().__init__(config)

        setattr(
            self, 'q_proj',
            nn.Conv2d(
                in_channels=self.config['n_embd'],
                out_channels=self.config['n_embd'],
                kernel_size=1,
                bias=self.config['bias'],
            )
        )

        setattr(
            self, 'k_proj',
            nn.Conv2d(
                in_channels=self.config['n_embd'],
                out_channels=self.config['n_embd'],
                kernel_size=1,
                bias=self.config['bias'],
            )
        )

        setattr(
            self, 'v_proj',
            nn.Conv2d(
                in_channels=self.config['n_embd'],
                out_channels=self.config['n_embd'],
                kernel_size=1,
                bias=self.config['bias'],
            )
        )

        setattr(
            self, 'out_proj',
            nn.Conv2d(
                in_channels=self.config['n_embd'],
                out_channels=self.config['n_embd'],
                kernel_size=1,
                bias=self.config['bias'],
            )
        )

        # override the bias buffer with a 4D tensor reshaped
        n_ctx = self.config['block_size']
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, n_ctx, 1, n_ctx))

    def forward(self, x, kv_cache : Optional[Dict] = None):
        """
        Arguments:
            x: torch.tensor, shape (batch_size, seq_len, n_embd)
                Input tensor
            kv_cache: Optional[Dict], default None
                Key-Value cache for fast inference
        """
        seq_len = x.size(1)

        q = self.q_proj(x)  # (batch_size, n_embd, 1, seq_len)
        k = self.k_proj(x)  # (batch_size, n_embd, 1, seq_len)
        v = self.v_proj(x)  # (batch_size, n_embd, 1, seq_len)

        dim_per_head = self.config['n_embd'] // self.config['n_head']
        mh_q = q.split(dim_per_head, dim=1)  # (batch_size, dim_per_head, 1, seq_len) * n_heads
        mh_k = k.split(dim_per_head, dim=1)  # (batch_size, dim_per_head, 1, seq_len) * n_heads
        mh_v = v.split(dim_per_head, dim=1)  # (batch_size, dim_per_head, 1, seq_len) * n_heads

        if kv_cache is not None:
            past_key = kv_cache[0]
            past_value = kv_cache[1]
            mh_k = [torch.cat((past_key, k), dim=-1) for k in mh_k]
            mh_v = [torch.cat((past_value, v), dim=-1) for v in mh_v]

        full_seq_len = k.size(-1)
        present = (mh_k, mh_v)

        normalize_factor = float(dim_per_head)**-0.5
        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * normalize_factor
            for qi, ki in zip(mh_q, mh_k)
        ]  # (batch_size, seq_len, 1, seq_len) * n_heads

        attn_weights = [
            aw.masked_fill(
                self.bias[:, full_seq_len-seq_len:full_seq_len, :, :full_seq_len] == 0,
                float('-inf')
            )
            for aw in attn_weights
        ]

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights]

        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]  # (batch_size, dim_per_head, 1, seq_len) * n_heads

        attn = torch.cat(attn, dim=1)  # (batch_size, n_embd, 1, seq_len)

        attn = self.out_proj(attn)  # (batch_size, n_embd, 1, seq_len)

        return attn, present


class NonCausalSelfAttentionANE(NonCausalSelfAttention):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        setattr(
            self, 'c_attn',
            nn.Conv2d(
                in_channels=self.config.n_embd,
                out_channels=self.config.n_embd * 3,
                kernel_size=1,
                bias=self.config.bias,
            )
        )

        setattr(
            self, 'c_proj',
            nn.Conv2d(
                in_channels=self.config.n_embd,
                out_channels=self.config.n_embd,
                kernel_size=1,
                bias=self.config.bias,
            )
        )

    def forward(self, x):
        """
        Arguments:
            x: torch.tensor, shape (batch_size, seq_len, n_embd)
                Input tensor
            kv_cache: Optional[Dict], default None
                Key-Value cache for fast inference
        """
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=1)  # (batch_size, n_embd, 1, seq_len) * 3

        dim_per_head = self.config.n_embd // self.config.n_head
        mh_q = q.split(dim_per_head, dim=1)  # (batch_size, dim_per_head, 1, seq_len) * n_heads
        mh_k = k.split(dim_per_head, dim=1)  # (batch_size, dim_per_head, 1, seq_len) * n_heads
        mh_v = v.split(dim_per_head, dim=1)  # (batch_size, dim_per_head, 1, seq_len) * n_heads

        normalize_factor = float(dim_per_head)**-0.5
        attn_weights = [
            torch.einsum('bchq,bchk->bkhq', [qi, ki]) * normalize_factor
            for qi, ki in zip(mh_q, mh_k)
        ]  # (batch_size, seq_len, 1, seq_len) * n_heads

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights]

        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]  # (batch_size, dim_per_head, 1, seq_len) * n_heads

        attn = torch.cat(attn, dim=1)  # (batch_size, n_embd, 1, seq_len)

        attn = self.c_proj(attn)  # (batch_size, n_embd, 1, seq_len)

        return attn


class MLPOptimizedForANE(MLP):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        setattr(
            self, 'c_fc',
            nn.Conv2d(
                in_channels=self.config.n_embd,
                out_channels=self.config.n_embd * 4,
                kernel_size=1,
                bias=self.config.bias,
            )
        )

        setattr(
            self, 'c_proj',
            nn.Conv2d(
                in_channels=self.config.n_embd * 4,
                out_channels=self.config.n_embd,
                kernel_size=1,
                bias=self.config.bias,
            )
        )


class BlockOptimizedForANE(Block):

    def __init__(self, config):
        super().__init__(config, None)

        setattr(self, 'ln_1', LayerNormANE(config.n_embd, config.bias, eps=EPS))
        setattr(self, 'attn', CausalSelfAttentionANE(config))
        setattr(self, 'ln_2', LayerNormANE(config.n_embd, config.bias, eps=EPS))
        setattr(self, 'mlp', MLPOptimizedForANE(config))

    def forward(self, tokens, kv_cache: Optional[Dict] = None):
        """
        Arguments
        ---------
        tokens : torch.tensor
            Input tensor

        kv_cache : Optional[Dict]
            Key-Value cache for fast inference
        """
        attn_output, prev_kv_cache = self.attn(self.ln_1(tokens), kv_cache=kv_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, prev_kv_cache


class FineBlockOptimizedForANE(Block):

    def __init__(self, config):
        super().__init__(config, None)

        setattr(self, 'ln_1', LayerNormANE(config.n_embd, config.bias, eps=EPS))
        setattr(self, 'attn', NonCausalSelfAttentionANE(config))
        setattr(self, 'ln_2', LayerNormANE(config.n_embd, config.bias, eps=EPS))
        setattr(self, 'mlp', MLPOptimizedForANE(config))

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.tensor
            Input tensor

        kv_cache : Optional[Dict]
            Key-Value cache for fast inference
        """
        import ipdb; ipdb.set_trace()
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTOptimizedForANE(GPT):

    def __init__(self, config, merge_context):
        super().__init__(config)

        setattr(
            self, 'transformer',
            nn.ModuleDict(dict(
                wte = nn.Embedding(self.config['input_vocab_size'], self.config['n_embd']),
                wpe = nn.Embedding(self.config['block_size'], self.config['n_embd']),
                blocks = nn.ModuleList([
                    BlockOptimizedForANE(config)
                    for _ in range(self.config['n_layer'])
                ]),
                ln_f = LayerNormANE(self.config['n_embd'], bias=self.config['bias'], eps=EPS)
            ))
        )

        setattr(
            self, 'lm_head',
            nn.Conv2d(
                in_channels=self.config['n_embd'],
                out_channels=self.config['output_vocab_size'],
                kernel_size=1,
                bias=False,
            )
        )

        self.merge_context = merge_context

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def forward(self, tokens):
        """
        Arguments
        ---------
        tokens : torch.tensor, shape (batch_size, seq_len)
            Input tensor
        """

        if self.merge_context:
            t = tokens.shape[1] - 256
            tok_emb = torch.cat([
                self.transformer.wte(tokens[:, :256]) + self.transformer.wte(tokens[:, 256:512]),
                self.transformer.wte(tokens[:, 512:])
            ], dim=1)
        else:
            tok_emb = self.transformer.wte(tokens)

        position_ids = torch.arange(0, t, dtype=torch.long).unsqueeze(0)   # device?
        pos_emb = self.transformer.wpe(position_ids)

        x = tok_emb + pos_emb

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x[:, [-1], :])

        return logits


class FineGPTOptimizedForANE(FineGPT):

    def __init__(self, config):
        super().__init__(config)

        setattr(
            self, 'transformer',
            nn.ModuleDict(dict(
                wtes = nn.ModuleList([
                    nn.Embedding(self.config.input_vocab_size, self.config.n_embd)
                    for _ in range(self.config.n_codes_total)
                ]),
                wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
                blocks = nn.ModuleList([
                    FineBlockOptimizedForANE(config)
                    for _ in range(self.config.n_layer)
                ]),
                ln_f = LayerNormANE(self.config.n_embd, self.config.bias, eps=EPS)
            ))
        )

        setattr(
            self, 'lm_heads',
            nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.config.n_embd,
                    out_channels=self.config.output_vocab_size,
                    kernel_size=1,
                    bias=False,
                )
                for _ in range(self.config.n_codes_given, self.config.n_codes_total)
            ])
        )

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

        # TODO: should I shift the weight matrices as in original implementation?

    def forward(self, tokens, codebook_idx=2):  # TODO: remove default value
        """
        Arguments
        ---------
        token : torch.tensor, (batch_size, seq_len, codes)
            Input tensor

        codebook_idx : int
            Index of the codebook
        """
        t = tokens.shape[1]
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)  

        tok_embs = [
            wte(tokens[:, :, i]).unsqueeze(-1) for i, wte in enumerate(self.transformer.wtes)
        ]   # (batch_size, seq_len, n_embd, 1) * n_codes_total
        tok_emb = torch.cat(tok_embs, dim=-1)  # (batch_size, seq_len, n_embd, n_codes_total)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb[:, :, :, :codebook_idx + 1].sum(dim=-1) + pos_emb

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_heads[codebook_idx - self.config.n_codes_given](x)
        return logits


def convert_encoder_to_coreml(hparams, model, quantize=False):
    model.eval()

    input_shape = (1, hparams.block_size, hparams.n_codes_total)
    input_data = torch.randint(hparams.block_size, size=(hparams.block_size, hparams.n_codes_total)) \
                      .unsqueeze(0)
    traced_model = torch.jit.trace(model, input_data)

    model = ct.convert(
        traced_model,
        convert_to=None if quantize else "mlprogram",
        inputs=[ct.TensorType(name="tokens", shape=input_shape)],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model


parser = argparse.ArgumentParser(description='Convert Bark to CoreML')
parser.add_argument('--model_dir', type=str, required=True, help='model path to convert (e.g. bark-small, bark-large)')
parser.add_argument("--use_f16", action="store_true", help="Use f16 precision")


if __name__ == "__main__":
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise ValueError(f"Model path {model_dir} does not exist")

    checkpoint = torch.load(model_dir / "pytorch_model.bin", map_location='cpu')
    config = json.load(open(model_dir / "config.json", "r"))

    hparams = config["fine_acoustics_config"]
    hparams = AttrDict(rename_keys(hparams))

    encoder = FineGPTOptimizedForANE(hparams).eval()
    state_dict = {
        rename_checkpoint_key(k): v for k, v in checkpoint.items()
        if "fine_acoustics" in k
    }
    encoder.load_state_dict(state_dict)

    encoder = convert_encoder_to_coreml(hparams, encoder, quantize=True)

    # for prefix in ["semantic", "coarse_acoustics", "fine_acoustics"]:
    #     hparams = config[f"{prefix}_config"]

    #     if prefix == "fine_acoustics":
    #         encoder = FineGPTOptimizedForANE(hparams)
    #     else:
    #         encoder = GPTOptimizedForANE(hparams, merge_context=prefix=="semantic")

    #     encoder = convert_encoder_to_coreml(hparams, encoder, quantize=args.quantize)
    #     encoder.save(f"models/coreml-bark-{prefix.split("_")[0]}.mlpackage")

    print("Done.")