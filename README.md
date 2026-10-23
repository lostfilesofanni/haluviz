# haluviz

A J-space visualizer with **hallucination detection** for the original
**Qwen-1.8B-Chat** (2023). Built on [jspace-viz](https://github.com/Festyve/jspace-viz)
and the Jacobian-lens method from Anthropic's
[global workspace paper](https://transformer-circuits.pub/2026/workspace/index.html).

Every cell of the grid shows what one layer of the model is *disposed to say*
at one token. Generated tokens are scored for hallucination from that same
grid and highlighted in red.

## How the detector works

For each token the model generates, three signals are read off the column that
produced it:

| signal | question it answers |
|---|---|
| `p_final` | how confident was the output layer in the token it chose |
| `commit_depth` | how deep into the network did that token become (and stay) top-1 — 0 = early, 1 = only at the output |
| `support` | the max probability the token had in the mid-depth "workspace" band (25–85% of layers) before being emitted |

`score = mean(1 - p_final, commit_depth, 1 - support)`. Tokens over the
threshold are flagged, except punctuation and tokens copied from the prompt.
Contiguous flagged tokens become spans. Code: `jspace_viz/hallucination.py`.

Example, logit-lens mode:

```
Q: What is the capital city of Palau?
A: The capital city of Palau is Nha Trang.        ← flagged: "Nha Trang" (0.84)
Q: Who won the Nobel Prize in Literature in 1907?
A: Ernest Hemingway won the Nobel Prize ...       ← flagged: "Ernest Heming" (0.89)
```

Both are wrong (Ngerulmud; Kipling). The detector has no access to the truth:
it only reads the model's internals. `scripts/eval_detector.py` measures how
well that works on `data/questions.json`.

### Measured (logit-lens mode, labels from the Gemini judge)

| set | answers | wrong | AUROC |
|---|---|---|---|
| PopQA, 100 sampled (19 refusals excluded) | 81 | 71 | 0.80 |
| hand set | 30 | 11 | — |
| pooled | 111 | 82 | **0.90** |

Calibrated risk shown in the UI (pooled): answer score < 0.70 → 8 % wrong,
0.70–0.75 → 36 %, 0.75–0.80 → 75 %, 0.80–0.85 → 91 %, ≥ 0.85 → 100 %.

Caveats: under the logit lens two of the three signals are near-constant
(early layers never agree with the output), so `p_final` does most of the
work until a J-lens is fitted; confident errors are invisible to the
detector by construction, which is what the **Verify** button (Gemini judge)
is for. Full details: `docs/IMPLEMENTATION.md`.

## Setup (Apple Silicon, 8 GB is enough)

```sh
uv venv --python 3.12 && uv pip install -e '.[fit]'
.venv/bin/python scripts/convert_gguf_to_hf.py     # ~2 min; downloads the f16 GGUF
.venv/bin/haluviz                                  # http://127.0.0.1:8321
```

`convert_gguf_to_hf.py` turns the community GGUF of Qwen-1.8B-Chat into a
native `Qwen2ForCausalLM` checkpoint (the architectures are identical), so no
2023-era remote code is needed. The server loads it with int8 weights
(`dtype="int8"` in `jspace_viz/presets.py`): fp16 weights swap on an 8 GB
machine; int8 does not.

A read with 32 generated tokens takes ~30 s on an M1.

The server holds ~2.4 GB of wired memory while it runs. Run it in the
foreground and Ctrl+C when done; it also exits by itself after 10 minutes
without a read (`--idle-timeout 0` to disable).

## J-lens vs logit lens

Without a fitted lens the server runs in **logit-lens** mode (raw residual
→ unembed). To get the real J-lens, fit one (hours, resumable):

```sh
.venv/bin/python scripts/fit_lens.py --preset qwen-1.8b-chat --n-prompts 20 --max-seq-len 64
```

It lands in `lenses/` and the server picks it up on restart. Run the fit, the
server, and the eval one at a time — each loads its own copy of the model.

## Evaluate the detector

```sh
.venv/bin/python scripts/eval_detector.py        # stop the server first
```

Runs every question in `data/questions.json`, labels the answer right/wrong by
substring match, and reports AUROC / precision / recall of the max token score.
Add your own questions there.

## Roadmap

- **Phase 1 (this):** detect and highlight.
- Phase 2: suggest a fix — DoLa-style layer-contrast re-decoding using the
  per-layer logits already computed, plus abstention when the score is high.
- Phase 3: J-space steering — inject the lens vector of the correct concept
  mid-network and regenerate.

## Credits

`jspace_viz/` is vendored from [Festyve/jspace-viz](https://github.com/Festyve/jspace-viz)
(Apache-2.0) with modifications; see `NOTICE`.
