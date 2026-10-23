"""Convert the Qwen-1 (2023) chat GGUF into a native transformers Qwen2 model.

Qwen-1's architecture is tensor-for-tensor the Qwen2 architecture (RMSNorm,
rotate-half RoPE, QKV with bias, SwiGLU, MHA), so the only work is renaming
llama.cpp tensors and splitting the fused QKV. This sidesteps Qwen-1's
2023-era remote code (tuple KV cache, tiktoken) entirely.

    python scripts/convert_gguf_to_hf.py                  # downloads the f16 GGUF
    python scripts/convert_gguf_to_hf.py --gguf path.gguf # e.g. an ollama blob
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parent.parent
GGUF_REPO = "mradermacher/Qwen-1_8B-Chat-GGUF"
GGUF_FILE = "Qwen-1_8B-Chat.f16.gguf"
TOKENIZER_ID = "Qwen/Qwen1.5-1.8B-Chat"  # same vocab + ChatML template, HF-native
OUT = ROOT / "models" / "qwen-1_8b-chat-hf"

RENAME = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
BLOCK = {  # llama.cpp suffix -> HF suffix, for blk.N.*
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",  # Qwen-1 mlp.w2 (silu branch)
    "ffn_up": "mlp.up_proj",  # Qwen-1 mlp.w1
    "ffn_down": "mlp.down_proj",
}


def convert(gguf_path: str) -> dict[str, torch.Tensor]:
    reader = GGUFReader(gguf_path)
    arch = str(bytes(reader.fields["general.architecture"].parts[-1]), "utf-8")
    assert arch == "qwen", f"expected a qwen (v1) GGUF, got {arch!r}"
    state: dict[str, torch.Tensor] = {}
    for t in reader.tensors:
        arr = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).to(torch.float16)
        if t.name in RENAME:
            state[RENAME[t.name]] = arr
            continue
        _, n, rest = t.name.split(".", 2)
        kind, param = rest.rsplit(".", 1)
        if kind == "attn_qkv":
            q, k, v = arr.chunk(3, dim=0)
            for name, part in (("q", q), ("k", k), ("v", v)):
                state[f"model.layers.{n}.self_attn.{name}_proj.{param}"] = part.contiguous()
        else:
            state[f"model.layers.{n}.{BLOCK[kind]}.{param}"] = arr
    return state, reader


def check_tokenizer(reader: GGUFReader, tok) -> None:
    field = reader.fields["tokenizer.ggml.tokens"]
    gguf_tokens = [str(bytes(field.parts[i]), "utf-8") for i in field.data]
    ids = list(range(0, 2000, 7)) + [151643, 151644, 151645]
    bad = [(i, gguf_tokens[i], tok.convert_ids_to_tokens(i)) for i in ids if gguf_tokens[i] != tok.convert_ids_to_tokens(i)]
    assert not bad, f"tokenizer vocab mismatch vs {TOKENIZER_ID}: {bad[:5]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", default=None, help="local GGUF path (default: download f16 from the Hub)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    gguf_path = args.gguf or hf_hub_download(GGUF_REPO, GGUF_FILE)
    print("reading", gguf_path)
    state, reader = convert(gguf_path)
    n_layers = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("model.layers."))
    d_model = state["model.embed_tokens.weight"].shape[1]
    inter = state["model.layers.0.mlp.gate_proj.weight"].shape[0]
    vocab = state["model.embed_tokens.weight"].shape[0]
    config = Qwen2Config(
        hidden_size=d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=inter,
        vocab_size=vocab,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_sliding_window=False,
        hidden_act="silu",
        bos_token_id=151643,
        eos_token_id=151645,
        dtype="float16",
    )
    print(f"layers={n_layers} d_model={d_model} inter={inter} vocab={vocab}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    check_tokenizer(reader, tok)
    tok.eos_token = "<|im_end|>"

    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    (Path(args.out) / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151645, 151643], "pad_token_id": 151643, "do_sample": False}, indent=2)
    )
    del state

    # Self-check: a real forward pass must produce a sane chat answer.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=16, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print("self-check answer:", text.strip())
    assert "Paris" in text, "conversion produced garbage; check tensor mapping"
    print("ok ->", args.out)


if __name__ == "__main__":
    sys.exit(main())
