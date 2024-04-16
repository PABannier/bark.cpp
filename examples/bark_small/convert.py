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


parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--vocab-path", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--use-f16", action="store_true")


def parse_hparams(hparams, outfile, use_f16, overwrite_bias):
    """Parse GPT hyperparameters."""
    outfile.write(struct.pack("i", hparams["n_layer"]))
    outfile.write(struct.pack("i", hparams["n_head"]))
    outfile.write(struct.pack("i", hparams["n_embd"]))
    outfile.write(struct.pack("i", hparams["block_size"]))

    bias = 1 if overwrite_bias else hparams["bias"]
    outfile.write(struct.pack("i", int(bias)))

    try:
        outfile.write(struct.pack("ii", hparams["vocab_size"], hparams["vocab_size"]))
    except KeyError:
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


def parse_model_weights(checkpoint, prefix, outfile, use_f16):
    """Load GPT model checkpoint (text, fine, coarse)."""
    num_tensors = len(checkpoint)
    outfile.write(struct.pack("i", num_tensors))

    # Filter out the variables that are not part of the current model with prefix
    keys = [k for k in checkpoint.keys() if prefix in k]

    for name in keys:
        var_data = checkpoint[name].squeeze().numpy()
        print(f"Processing variable: {name} with shape: {var_data.shape}")

        n_dims = len(var_data.shape)

        # Remove prefix from the variable name and the dot
        name = name.replace(prefix + ".", "")

        if "lm_heads" in name:
            name = ".".join(name.split(".")[1:])
        else:
            name = ".".join(name.split(".")[2:])

        # rename headers to keep compatibility
        if name == "layernorm_final.weight":
            name = "model/ln_f/g"
        elif name == "layernorm_final.bias":
            name = "model/ln_f/b"
        elif name == "input_embeds_layer.weight":
            name = "model/wte/0"
        elif name == "position_embeds_layer.weight":
            name = "model/wpe"
        elif name == "lm_head.weight":
            name = "model/lm_head/0"
        elif re.match(r"wtes\.\d+\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/wte/{i}"
        elif re.match(r"layers\.\d+\.layernorm_1\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/g"
        elif re.match(r"layers\.\d+\.layernorm_1\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/b"
        elif re.match(r"layers\.\d+\.attn\.c_attn\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"layers\.\d+\.attn\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/b"
        elif re.match(r"h\.\d+\.attn\.c_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/w"
        elif re.match(r"h.\d+.attn.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/b"
        elif re.match(r"h.\d+.ln_2.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/g"
        elif re.match(r"h.\d+.ln_2.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/b"
        elif re.match(r"h.\d+.mlp.c_fc.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/w"
        elif re.match(r"h.\d+.mlp.c_fc.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/b"
        elif re.match(r"h.\d+.mlp.c_proj.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/w"
        elif re.match(r"h.\d+.mlp.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/b"
        elif re.match(r"lm_heads\.\d+\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/lm_head/{i}"
        else:
            print(f"Unrecognized variable name: {name}")

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


def generate_file(dir_model, fout, use_f16, overwrite_bias=False):
    checkpoint = torch.load(dir_model / "pytorch_model.bin", map_location="cpu")
    config = json.load(open(dir_model / "config.json", "r"))
    for prefix in ["semantic", "coarse_acoustics", "fine_acoustics", "codec_model"]:
        parse_hparams(config, prefix, fout, use_f16, overwrite_bias)
        parse_model_weights(checkpoint, prefix, fout, use_f16)


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
    vocab_path = Path(args.vocab_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    out_file = out_dir / "ggml_weights.bin"

    # Write magic number
    fout = open(out_file, "wb")
    fout.write(struct.pack("i", 0x67676d6c))

    generate_vocab_file(vocab_path, fout)
    print(" Vocab written.")

    generate_file(dir_model, fout, args.use_f16)
    print(" Model written.")

    fout.close()

    print("Done.")
