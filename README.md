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

### Measured (logit-lens mode, 30 questions, 10 answered wrong)

Question-level: max token score vs. "answer was wrong".

| | |
|---|---|
| AUROC | **0.915** |
| threshold 0.85 (default) | precision 0.53, recall 0.80 |
| threshold 0.90 | precision 0.88, recall 0.70 |

Caveats, honestly:
- Under the logit lens, `commit_depth` ≈ 1 and `support` ≈ 0 for almost every
  token (early layers never agree with the output), so the score floors near
  0.65 and `p_final` does the separating. Those two signals only become
  informative with a fitted J-lens.
- Several "correct" answers that score high are long answers whose *extra*
  claims are dubious (e.g. "published in 1934" for The Master and Margarita).
  The eval only checks the gold entity, so some false positives are real
  hallucinations the labels don't see.
- 30 questions is a small set. Add yours to `data/questions.json` and re-run.

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
