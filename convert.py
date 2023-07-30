"""Convert Bark's GPT and Encodec checkpoints into the GGML format.

The bytes are packed in a binary file in the following order:
    - Magic (`ggml` in binary format)
    - Tensors

For each tensor, the bytes are packed as follows:
    - Number of dimensions    (int)
    - Name length             (int)
    - Dimensions              (int[n_dims])
    - Name                    (char[name_length])
    - Data                    (float[n_dims])

Note
----
Encodec uses weight normalization for its convolutional layers. All the weights are
decomposed into two tensors called with the suffixes _weight_v and _weight_g. A simple
call to the hook torch._weight_norm allows to get the final weight tensor of the
convolution from weight_v and weight_g. To drastically reduce the number of operations
at inference time, the ggml weights file only contain the final convolution weights but
does not store the decomposition into weight_v and weight_g.

Example
-------
```bash
    python convert.py \
        --dir-model ~/.cache/suno/bark_v0 \
        --codec-path ~/Documents/encodec.cpp/ggml_weights \
        --vocab-path ./ggml_weights/ \
        --out-dir ./ggml_weights/
```
"""
import argparse
from pathlib import Path
import re
import struct

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--codec-path", type=str, required=True)
parser.add_argument("--vocab-path", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)


def parse_codec_model(checkpoint, out_dir):
    """Load encodec model checkpoint."""
    outfile = open(out_dir, "wb")
    outfile.write(struct.pack("i", 0x67676d6c))  # ggml magic

    for name in checkpoint.keys():
        if "encoder." in name:
            # bark only uses Encodec's quantizer and decoder.
            continue

        if "weight_g" in name:
            # the tensor has already been parsed with the corresponding "weight_v"
            # tensor to form the final weights tensor of the convolution, therefore
            # we skip it
            continue

        if "inited" in name or "cluster_size" in name or "embed_avg" in name:
            # "inited", "cluster_size" and "embed_avg" tensors in quantizer are not used
            # for the forward pass
            continue

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

        print(f"Processing variable: {name} with shape: {var_data.shape}")

        if var_data.dtype != np.float32:
            print("  Converting to float32")
            var_data = var_data.astype(np.float32)

        n_dims = len(var_data.shape)
        encoded_name = name.encode("utf-8")
        ftype = 0  # float32
        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype))

        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        outfile.write(encoded_name)

        var_data.tofile(outfile)

    outfile.close()

def parse_hparams(hparams, outfile):
    """Parse GPT hyperparameters."""
    outfile.write(struct.pack("i", hparams["n_layer"]))
    outfile.write(struct.pack("i", hparams["n_head"]))
    outfile.write(struct.pack("i", hparams["n_embd"]))
    outfile.write(struct.pack("i", hparams["block_size"]))

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

    outfile.write(struct.pack("ii", n_lm_heads, n_wtes))

def parse_text_models(checkpoint, outfile):
    """Load GPT model checkpoint (text, fine, coarse)."""
    for name in checkpoint.keys():
        var_data = checkpoint[name].squeeze().numpy()
        print(f"Processing variable: {name} with shape: {var_data.shape}")

        n_dims = len(var_data.shape)

        ftype_cur = 0
        if var_data.dtype != np.float32:
            print("  Converting to float32")
            var_data = var_data.astype(np.float32)
            ftype_cur = 0

        # strip `_orig_mod.transformer.` prefix
        if name == "_orig_mod.lm_head.weight":
            name = "lm_head.weight"
        elif "lm_heads" in name:
            name = ".".join(name.split(".")[1:])
        else:
            name = ".".join(name.split(".")[2:])

        # rename headers to keep compatibility
        if name == "ln_f.weight":
            name = "model/ln_f/g"
        elif name == "ln_f.bias":
            name = "model/ln_f/b"
        elif name == "wte.weight":
            name = "model/wte/0"
        elif name == "wpe.weight":
            name = "model/wpe"
        elif name == "lm_head.weight":
            name = "model/lm_head/0"
        elif re.match(r"wtes\.\d+\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/wte/{i}"
        elif re.match(r"h\.\d+\.ln_1\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/g"
        elif re.match(r"h\.\d+\.ln_1\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/b"
        elif re.match(r"h\.\d+\.attn\.c_attn\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"h\.\d+\.attn\.c_attn\.bias", name):
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

        encoded_name = name.encode("utf-8")

        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))
        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        outfile.write(encoded_name)

        var_data.tofile(outfile)

def generate_file(in_file, out_dir):
    with open(out_dir, "wb") as fout:
        fout.write(struct.pack("i", 0x67676d6c))  # ggml magic

        checkpoint = torch.load(in_file, map_location="cpu")
        parse_hparams(checkpoint["model_args"], fout)
        parse_text_models(checkpoint["model"], fout)

def generate_vocab_file(dir_model, out_dir):
    """Parse vocabulary."""
    # Even if bark relies on GPT to encode text, it uses BertTokenizer (WordPiece)
    with open(dir_model / "vocab.txt", "r", encoding="utf-8") as fin:
        vocab = fin.readlines()

    with open(out_dir, "wb") as fout:
        fout.write(struct.pack("i", 0x67676d6c))  # ggml magic
        fout.write(struct.pack("i", len(vocab)))
        print("Vocab size:", len(vocab))

        for token in vocab:
            data = bytearray(token[:-1], "utf-8")  # strip newline at the end
            fout.write(struct.pack("i", len(data)))
            fout.write(data)


if __name__ == "__main__":
    args = parser.parse_args()

    dir_model = Path(args.dir_model)
    codec_path = Path(args.codec_path)
    vocab_path = Path(args.vocab_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    generate_vocab_file(vocab_path, out_dir / "ggml_vocab.bin")
    print(" Vocab loaded.")

    generate_file(dir_model / "text_2.pt", out_dir / "ggml_weights_text.bin")
    print(" Text model loaded.")

    generate_file(dir_model / "coarse_2.pt", out_dir / "ggml_weights_coarse.bin")
    print(" Coarse model loaded.")

    generate_file(dir_model / "fine_2.pt", out_dir / "ggml_weights_fine.bin")
    print(" Fine model loaded.")

    codec_chkpt = torch.load(codec_path / "encodec_24khz-d7cc33bc.th", map_location="cpu")
    parse_codec_model(codec_chkpt, out_dir / "ggml_weights_codec.bin")
    print(" Codec model loaded.")

    print("Done.")
