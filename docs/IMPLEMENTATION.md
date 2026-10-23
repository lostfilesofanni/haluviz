# haluviz — implementation notes

Working documentation of what was built, why, and what was measured. Kept
current as phases land. Dates are 2026.

## 1. Goal

Show, for a small open chat model, *what the model is thinking at every layer
while it answers*, and use those internals to **detect where the answer is
hallucinated**, without access to the truth. Later phases: verify against an
external judge, and suggest a fix.

Target model: the original **Qwen-1.8B-Chat** (Nov 2023). Chosen deliberately
as an old, small, visibly unreliable model.

## 2. Background in plain terms

A transformer answers a question by pushing a vector per token through 24
layers. The vector at layer 24 is turned into word probabilities by the
*unembedding* matrix. The vectors at layers 1–23 are not meant to be read that
way, but two tricks let us try:

- **Logit lens.** Push the layer-L vector straight through the unembedding as
  if the network ended there. Works for the last few layers; for the middle it
  mostly yields noise (for a bilingual model, Chinese noise).
- **Jacobian lens (J-lens).** Anthropic, *Verbalizable Representations Form a
  Global Workspace in Language Models* (transformer-circuits.pub/2026/workspace).
  For each layer L, fit one matrix `J_L = E[∂h_final / ∂h_L]`: the average
  effect the rest of the network has on a layer-L vector. Read
  `unembed(J_L · h_L)` instead. Now the middle layers read as real words: the
  concepts the model is "holding in mind" before it speaks. The sparse set of
  such words is the **J-space**; the paper argues it behaves like a global
  workspace (small, reportable, causally used for multi-step reasoning).

The method is model-agnostic; Anthropic released reference code
(github.com/anthropics/jacobian-lens) and Neuronpedia publishes fitted lenses
for Qwen3, Gemma, Llama, etc. None exists for Qwen-1, so we fit our own.

**Hallucination hypothesis.** A fact the model knows should appear in the
workspace several layers before it is spoken and stay top-1; a fabrication
appears only at the output. The detector is built around that.

## 3. Architecture

```
GGUF (community, f16) ──convert──▶ models/qwen-1_8b-chat-hf  (native Qwen2ForCausalLM)
                                            │
                     jspace_viz/server.py (FastAPI, one model in memory, int8)
                       POST /api/read   → read_grid()  → grid + hallucination signals
                       POST /api/verify → verify.judge() (Gemini)
                       GET  /api/info
                                            │
                     jspace_viz/static/  (vanilla JS, no build) — grid, workspace
                       chips, hallucination panel, threshold slider, verify button
```

`jspace_viz/` is vendored from Festyve/jspace-viz (Apache-2.0) and modified;
`NOTICE` records the provenance.

## 4. Getting the model to run at all

### 4.1 Why not just load Qwen-1 from HuggingFace
- `Qwen/Qwen-1_8B-Chat` (bf16) now returns 401; only the base model and
  CUDA-only GPTQ quants remain.
- Its modeling code is 2023-era remote code: tuple KV cache, tiktoken
  tokenizer, `trust_remote_code`. Broken on current transformers, and
  jspace-viz refuses remote code.
- Serving it through ollama/llama.cpp is not an option either: those expose
  text only, and we need the residual stream at every layer.

### 4.2 The conversion (`scripts/convert_gguf_to_hf.py`)
Observation: Qwen-1's architecture is tensor-for-tensor the **Qwen2**
architecture that transformers supports natively (RMSNorm, rotate-half RoPE,
QKV projections *with* bias, SwiGLU, 16 heads = 16 KV heads). So:

1. Download the lossless f16 GGUF (`mradermacher/Qwen-1_8B-Chat-GGUF`).
2. Dequantize every tensor with `gguf-py`.
3. Rename llama.cpp names → Qwen2 names; split the fused `attn_qkv` into
   q/k/v; `ffn_gate`=`mlp.w2`, `ffn_up`=`mlp.w1`. No RoPE permutation (both
   use NEOX/rotate-half).
4. `Qwen2Config(hidden=2048, layers=24, heads=16, kv_heads=16,
   intermediate=5504, vocab=151936, rope_theta=1e4, rms_eps=1e-6, untied head)`.
   Note intermediate is **5504**, not the 11008 in Qwen-1's config, because
   Qwen-1 splits that value in half internally.
5. Tokenizer: Qwen1.5's HF-native tokenizer has the same vocabulary and ChatML
   template; the script asserts token-id equality against the GGUF's embedded
   vocab before using it.
6. Self-check: greedy-generate "What is the capital of France?" and assert
   "Paris".

Result: a standard checkpoint, no remote code, works on MPS/CUDA/CPU.

### 4.3 Memory engineering (the dominant problem)
Dev machine: Apple M1, **8 GB RAM**, already 9–14 GB into swap from other apps.

| configuration | forward pass (30 tok) | 32-token generation |
|---|---|---|
| fp16 weights on MPS | 25 s (pure paging; 4 layers took 5–8 s each) | >200 s |
| **int8 weight-only** (`Int8Linear`, per-row absmax scale, dequantized per call) | ~1 s | ~13 s |

`jspace_viz/model.py::Int8Linear` halves resident weight memory (2.4 GB total
with fp16 embed/lm_head). Same precision class as the Q8_0 GGUF the model was
originally tested with. Gradients still flow to activations, so lens fitting
works through it (but doubles memory during backward, hence fitting on Colab).

Other server changes forced by the machine:
- **Model lock.** FastAPI runs sync endpoints in a thread pool; two concurrent
  reads on MPS crash the process with a Metal command-buffer assertion. All
  model use is serialized (`_model_lock`).
- **Idle exit.** The server exits after N idle minutes (default 10) so a
  forgotten process doesn't hold ~2.4 GB wired.
- **Optional lens.** Without a fitted lens the server runs in logit-lens mode
  instead of refusing to start.

A full read (32 new tokens + 24-layer readout) is ~30 s on this machine.
Breakdown when uncontended: generation ~13 s (int8 dequant each step; no
native int8 kernels in PyTorch/MPS), hooked forward ~1 s, 24 × (unembed over
151,936 vocab + softmax + top-k) ~2.5 s, remainder paging.

## 5. The hallucination detector (`jspace_viz/hallucination.py`)

For each **generated** token, read the column that produced it (position t−1
predicts token t) across layers:

| signal | definition | intuition |
|---|---|---|
| `p_final` | probability the output layer gave the emitted token | confidence |
| `commit_depth` | fraction of depth at which the token became top-1 *and stayed* top-1 through the output (0 = early, 1 = only at output) | decided late = guessed |
| `support` | max probability the token had in the mid-depth band (25–85 % of layers) | was it "in mind" before it was said |

`score = mean(1 − p_final, commit_depth, 1 − support)` ∈ [0, 1].

Rules learned from data:
- Punctuation/whitespace tokens are never flagged.
- Tokens **copied from the prompt** are never flagged (copying ≠ hallucinating).
- Flags apply to **whole words**: if the sub-word "ha" in " N|ha| Tr|ang"
  crosses the line, "Nha" is flagged, not "ha".
- Stop at the end-of-turn token.
- Contiguous flagged words form spans.

Per-layer `next_prob` (probability of the actual next token) was added to the
grid rows in `analysis.py::read_grid` to make `support` and `p_final` exact
rather than top-k approximations.

### 5.1 Calibration on the hand-labeled set (`data/questions.json`, 30 Q)
Logit-lens mode, question-level metric = max token score vs. "answer wrong":

| | |
|---|---|
| wrong answers | 10 / 30 |
| AUROC | **0.915** |
| threshold 0.80 (default) | precision 0.56, recall 1.00 |
| threshold 0.85 | precision 0.53, recall 0.80 |
| threshold 0.90 | precision 0.88, recall 0.70 |

Findings:
- Under the logit lens `commit_depth ≈ 1` and `support ≈ 0` for nearly every
  content token (early layers never agree with the output), so the score has a
  floor near 0.65 and `p_final` does the separating. These two signals should
  only become informative with the J-lens — this is the central experiment
  still to run.
- Several "correct" answers that score high are long answers whose extra
  claims are dubious (e.g. "published in 1934" for *The Master and Margarita*).
  The label only checks the gold entity.
- Examples caught: "capital of Palau is **Nha Trang**" (0.84), "1907 Nobel in
  Literature … **Fyodor Dostoevsky**" (0.90), "first woman Fields Medal … **Ada
  Lovelace**" (0.95), "smallest South American country … **Chile**" (0.96).

### 5.2 Why a binary flag is the wrong output
"Possible hallucination" fires on correct-but-hesitant answers (rare names,
numbers). The detector measures *guessing*, not *falsehood*; those correlate
(AUROC 0.9) but aren't equal. The planned fix (§7, in progress) is a
calibrated probability instead of a threshold.

## 6. Verifier (`jspace_viz/verify.py`, `POST /api/verify`)

The internals can't see the truth. The verifier asks a stronger model
(Gemini, REST API, key from `.env`) to judge `question + answer` and return
`{verdict: correct|wrong|unsure, correct_answer, explanation}`. Used for:
- the **Verify answer** button in the UI;
- auto-labeling free-form answers in evaluations.

Failure mode is explicit: any API error returns `unsure` with the error text;
a read never fails because the verifier did.

## 7. Evaluation without hand labels (`scripts/eval_popqa.py`)

PopQA (akariasai/PopQA, 14,267 long-tail factual questions with gold aliases)
is the calibration set. Answer = correct iff any alias appears in the
continuation. The script writes:
- `data/popqa_results.json`: per-question rows incl. per-token signals;
- `data/calibration.json`: score bins → empirical P(wrong), monotone-smoothed.

The UI will use the bins to show *"estimated chance this answer is wrong"*
with three bands instead of a bare flag. *(Run in progress at time of
writing; numbers to be added.)*

## 8. UI

- Layer × position grid; blue intensity = confidence; hover for top-k,
  entropy, kurtosis. Click to pin a token and view its rank at every cell.
- **english only** toggle: show the highest-ranked English token per cell
  (hides bilingual-noise cells of the logit lens).
- Workspace chips: mid-band concepts not in the prompt.
- Hallucination panel: answer with flagged words marked, spans with the three
  signals, threshold slider (re-flags client-side), Verify button.
- Generated columns are green; the page scrolls to them after a read.
- `live` (re-read while typing) off by default: each read is ~20–30 s here.

## 9. Fitting the J-lens (`colab/fit_lens_colab.ipynb`)

Fitting = for each layer, average over a text corpus of
`∂h_final/∂h_L`, estimated with one-hot cotangents: ~`d_model / dim_batch`
backward passes per prompt. The paper used 1000 prompts; the jspace-viz author
got a usable lens for a same-sized model from 40; ~10 already beats the logit
lens.

On the 8 GB M1 the attempt was killed by the OS for memory. The notebook does
it on a free Colab T4 in fp32: ~1 min/prompt, 30 prompts, checkpointed to
Google Drive. Output drops into `lenses/` and the server picks it up.

## 10. Roadmap

- **Phase 1 (done):** visualizer + detector + hand-labeled calibration.
- **Phase 1b (in progress):** J-lens fit; PopQA calibration; calibrated
  probability display; verifier.
- **Phase 2 — suggest a fix:**
  - *Abstain*: when the risk is high, show "I'm not sure" as the better answer.
  - *DoLa-style re-decoding*: prefer tokens whose probability *rises* between a
    middle layer and the output (knowledge emerging) over tokens that only
    appear at the end (surface guessing). We already compute per-layer logits.
    Show original vs re-decoded side by side.
  - *Verified correction*: show the judge's correct answer next to the span.
- **Phase 3 — steering:** with the J-lens, add the lens direction of the
  correct concept into the middle layers and regenerate; test whether the
  hallucination was a retrieval failure the workspace can repair (the paper's
  causal claim, reproduced on a tiny 2023 model).

## 11. Known limitations

- Logit-lens signals 2 and 3 are near-constant; J-lens results pending.
- 30-question hand set is small; PopQA run addresses this.
- Substring labels miss paraphrases; the judge addresses this.
- Speed is bounded by the machine, not the code.
- The J-lens only reads single-token concepts; multi-token entities show up as
  their first piece.

## 12. Decisions log

| date | decision | why |
|---|---|---|
| 10-22 | Keep Qwen-1 instead of Qwen1.5/Qwen3 | user's choice; it's the model they tested |
| 10-22 | Convert GGUF → native Qwen2 rather than fight remote code | identical architecture; zero compat debt |
| 10-22 | Vendor jspace-viz rather than write a visualizer | Apache-2.0, exactly the needed UI, validated lens math |
| 10-22 | int8 weight-only | fp16 paged at 25 s/forward on 8 GB |
| 10-22 | Default threshold 0.80 | full recall on the hand set; slider for precision |
| 10-23 | Fit lens on Colab | OS killed the local fit for memory |
| 10-23 | PopQA for calibration | gold aliases → no hand labels |
| 10-23 | Gemini as judge | user-provided key; local 3B judge too weak and memory-contended |
