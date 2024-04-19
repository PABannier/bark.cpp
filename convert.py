"""Convert Bark's GPT and Encodec checkpoints into the GGML format.

The file is structured as follows:
    - Hyperparameters
    - Vocabulary
    - Text model
    - Coarse model
    - Fine model

The bytes are packed in a binary file in the following order:
    - Magic (`ggml` in binary format)
    - Tensors

For each tensor, the bytes are packed as follows:
    - Number of dimensions    (int)
    - Name length             (int)
    - Dimensions              (int[n_dims])
    - Name                    (char[name_length])
    - Data                    (float[n_dims])

Example
-------
```bash
    python convert.py \
        --dir-model ~/.cache/suno/bark_v0 \
        --vocab-path ./ggml_weights/ \
        --out-dir ./ggml_weights/ \
        --use-f16
```
"""
import argparse
from pathlib import Path
import re
import struct
import json

import numpy as np
import torch


DECODER_CONV_TRANSPOSE_LAYERS = [
    "decoder.layers.3.conv.bias",
    "decoder.layers.3.conv.weight",
    "decoder.layers.6.conv.bias",
    "decoder.layers.6.conv.weight",
    "decoder.layers.9.conv.bias",
    "decoder.layers.9.conv.weight",
    "decoder.layers.12.conv.bias",
    "decoder.layers.12.conv.weight",
]


parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=False)
parser.add_argument("--use-f16", action="store_true")


def parse_codec_hparams(config, outfile, use_f16):
    """Parse Encodec hyperparameters."""
    in_channels = config["audio_channels"]
    hidden_dim = config["hidden_size"]
    n_filters = config["num_filters"]
    kernel_size = config["kernel_size"]
    residual_kernel_size = config["residual_kernel_size"]
    n_bins = config["codebook_size"]
    bandwidth = 24   # TODO: hardcoded
    sr = config["sampling_rate"]
    ftype = int(use_f16)

    outfile.write(struct.pack("i", in_channels))
    outfile.write(struct.pack("i", hidden_dim))
    outfile.write(struct.pack("i", n_filters))
    outfile.write(struct.pack("i", kernel_size))
    outfile.write(struct.pack("i", residual_kernel_size))
    outfile.write(struct.pack("i", n_bins))
    outfile.write(struct.pack("i", bandwidth))
    outfile.write(struct.pack("i", sr))
    outfile.write(struct.pack("i", ftype))


def parse_hparams(config, prefix, outfile, use_f16):
    """Parse GPT hyperparameters."""
    hparams = config[f"{prefix}_config"]

    outfile.write(struct.pack("i", hparams["num_layers"]))
    outfile.write(struct.pack("i", hparams["num_heads"]))
    outfile.write(struct.pack("i", hparams["hidden_size"]))
    outfile.write(struct.pack("i", hparams["block_size"]))

    # trick: for fine model, we need to set the bias flag to true, since there are
    # bias for the layer norm (to refactor)
    bias = True if prefix == "fine_acoustics" else hparams["bias"]
    outfile.write(struct.pack("i", int(bias)))

    outfile.write(
        struct.pack("ii", hparams["input_vocab_size"], hparams["output_vocab_size"])
    )

    n_lm_heads, n_wtes = None, None
    try:
        # only for fine text model
        n_lm_heads = hparams["n_codes_total"] - hparams["n_codes_given"]
        n_wtes = hparams["n_codes_total"]
    except KeyError:
        n_lm_heads, n_wtes = 1, 1

    ftype = int(use_f16)

    outfile.write(struct.pack("iii", n_lm_heads, n_wtes, ftype))


def parse_codec_model_weights(checkpoint, outfile, use_f16):
    """Load encodec model checkpoint."""
    keys = [k for k in checkpoint.keys() if "codec_model" in k]

    for name in keys:
        if "weight_g" in name:
            # the tensor has already been parsed with the corresponding "weight_v"
            # tensor to form the final weights tensor of the convolution, therefore
            # we skip it
            continue

        if "inited" in name or "cluster_size" in name or "embed_avg" in name:
            # "inited", "cluster_size" and "embed_avg" tensors in quantizer are not used
            # for the forward pass
            continue

        # Remove prefix from the variable name and the dot
        clean_name = name.replace("codec_model.", "")

        var_data = checkpoint[name]

        if not "weight_v" in name:
            # if conv kernel, do not squeeze because 3d tensor
            var_data = var_data.numpy().squeeze()
        else:
            # weight_v has its corresponding magnitude tensor to rescale the weights
            # of the convolutional layers. We parse both kinds of weights jointly to
            # build the final weight tensor of the convolution.
            base_name = name.split(".")[:-1]
            weight_g_name = ".".join(base_name + ["weight_g"])
            var_data_g = checkpoint[weight_g_name]

            final_var_data = torch._weight_norm(var_data, var_data_g, dim=0)
            var_data = final_var_data.numpy()

            name = ".".join(base_name + ["weight"])
            clean_name = name.replace("codec_model.", "")

        if "encoder" in clean_name or "decoder" in clean_name:
            if clean_name in DECODER_CONV_TRANSPOSE_LAYERS:
                pattern = r"decoder.layers.(\d+).conv\.(bias|weight)$"
                replacement = r"decoder.model.\1.convtr.convtr.\2"
                clean_name = re.sub(pattern, replacement, clean_name)
            elif "conv" in clean_name:
                pattern = r"(encoder|decoder).layers.(\d+)(.*?).conv\.(bias|weight)$"
                replacement = r"\1.model.\2\3.conv.conv.\4"
                clean_name = re.sub(pattern, replacement, clean_name)
            elif "lstm" in clean_name:
                clean_name = clean_name.replace("layers", "model")
        elif "quantizer" in clean_name:
            pattern = r"quantizer.layers.(\d+)\.codebook\.(.+)$"
            replacement = r"quantizer.vq.layers.\1._codebook.\2"
            clean_name = re.sub(pattern, replacement, clean_name)
        else:
            raise Exception(f"Unrecognized variable name: {clean_name}")

        print(f"Processing variable: {name} with shape: {var_data.shape}")

        if use_f16:
            if "embed" in name:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0
            elif "weight" in name:
                print("  Converting to float16")
                var_data = var_data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0
        else:
            print("  Converting to float32")
            var_data = var_data.astype(np.float32)
            ftype_cur = 0

        n_dims = len(var_data.shape)
        encoded_name = clean_name.encode("utf-8")
        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))

        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        outfile.write(encoded_name)

        var_data.tofile(outfile)

    outfile.close()


def parse_model_weights(checkpoint, prefix, outfile, use_f16):
    """Load GPT model checkpoint (text, fine, coarse)."""
    keys = [k for k in checkpoint.keys() if prefix in k]
    keys = [k for k in keys if "attn.bias" not in k]

    num_tensors = len(keys)
    outfile.write(struct.pack("i", num_tensors))

    # Filter out the variables that are not part of the current model with prefix

    for name in keys:
        var_data = checkpoint[name].squeeze().numpy()
        print(f"Processing variable: {name} with shape: {var_data.shape}")

        n_dims = len(var_data.shape)

        # Remove prefix from the variable name and the dot
        name = name.replace(prefix + ".", "")

        # rename headers to keep compatibility
        if name == "layernorm_final.weight":
            name = "model/ln_f/g"
        elif name == "layernorm_final.bias":
            name = "model/ln_f/b"
        elif name == "input_embeds_layer.weight":
            name = "model/wte/0"
        elif re.match(r"input_embeds_layers\.\d+\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/wte/{i}"
        elif name == "position_embeds_layer.weight":
            name = "model/wpe"
        elif name == "lm_head.weight":
            name = "model/lm_head/0"
        elif re.match(r"layers\.\d+\.layernorm_1\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/g"
        elif re.match(r"layers\.\d+\.layernorm_1\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/b"
        elif re.match(r"layers\.\d+\.layernorm_2\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/g"
        elif re.match(r"layers\.\d+\.layernorm_2\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/b"
        elif re.match(r"layers\.\d+\.attn\.bias", name):
            # this pattern is the lower triangular matrix of the attention bias
            # we do not need to load it
            continue
        elif re.match(r"layers\.\d+\.attn\.att_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"layers\.\d+\.attn\.out_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/w"
        elif re.match(r"layers\.\d+\.mlp\.in_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/w"
        elif re.match(r"layers\.\d+\.mlp\.out_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/w"
        elif re.match(r"lm_heads\.\d+\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/lm_head/{i}"
        else:
            raise Exception(f"Unrecognized variable name: {name}")

        if use_f16:
            if (name[-2:] == "/w" or "wte" in name or "lm_head" in name) and n_dims == 2:
                print("  Converting to float16")
                var_data = var_data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0
        else:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0

        encoded_name = name.encode("utf-8")

        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))
        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        outfile.write(encoded_name)

        var_data.tofile(outfile)


def generate_file(dir_model, fout, use_f16):
    checkpoint = torch.load(dir_model / "pytorch_model.bin", map_location="cpu")
    config = json.load(open(dir_model / "config.json", "r"))

    # Parse transformer hyperparameters and weights
    for prefix in ["semantic", "coarse_acoustics", "fine_acoustics"]:
        parse_hparams(config, prefix, fout, use_f16)
        parse_model_weights(checkpoint, prefix, fout, use_f16)

    # New model (Encodec.cpp expects the magic number) => re-write it
    fout.write(struct.pack("i", 0x67676d6c))

    # Parse neural codec weights
    parse_codec_hparams(config["codec_config"], fout, use_f16)
    parse_codec_model_weights(checkpoint, fout, use_f16)


def generate_vocab_file(dir_model, fout):
    """Parse vocabulary."""
    # Even if bark relies on GPT to encode text, it uses BertTokenizer (WordPiece)
    with open(dir_model / "vocab.txt", "r", encoding="utf-8") as fin:
        vocab = fin.readlines()

    fout.write(struct.pack("i", len(vocab)))
    print("Vocab size:", len(vocab))

    for token in vocab:
        data = bytearray(token[:-1], "utf-8")  # strip newline at the end
        fout.write(struct.pack("i", len(data)))
        fout.write(data)


if __name__ == "__main__":
    args = parser.parse_args()

    dir_model = Path(args.dir_model)
    if not dir_model.exists():
        raise ValueError(f"Could not find directory {dir_model}")

    if args.out_dir is None:
        out_dir = dir_model
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

    out_file = out_dir / "ggml_weights.bin"

    # Write magic number
    fout = open(out_file, "wb")
    fout.write(struct.pack("i", 0x67676d6c))

    generate_vocab_file(dir_model, fout)
    print(" Vocab written.")

    generate_file(dir_model, fout, args.use_f16)
    print(" Model written.")

    fout.close()

    print("Done.")
