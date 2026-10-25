# haluviz FAQ

Answers to the questions people ask about this project, from "how do I run
it" to "how is the number on the gauge computed". The long-form record of
every decision and measurement is in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## The project

**What is haluviz?**
A local web app that runs Qwen-1.8B-Chat, shows what each of its 24 layers is
"about to say" for every word it writes, and uses those internal signals to
estimate how likely the answer is to be made up. A stronger model (Gemini) can
then check the truth and suggest a fix.

**Why that model?**
It is small enough for an 8 GB laptop and old enough (Nov 2023) to get many
facts wrong, which makes hallucinations easy to find and watch.

**Where does the layer-reading method come from?**
The Jacobian lens (J-lens) from Anthropic's 2026 global workspace paper.
The visualizer started as a fork of Festyve/jspace-viz (Apache-2.0); the
`NOTICE` file records the provenance.

**What is in the repo?**

| path | role |
|---|---|
| `jspace_viz/server.py` | FastAPI server, one model in memory |
| `jspace_viz/analysis.py` | `read_grid`: runs the model, reads every layer |
| `jspace_viz/hallucination.py` | per-word signals, score, flags |
| `jspace_viz/lens.py`, `fitting.py` | load and fit a J-lens |
| `jspace_viz/fix.py` | abstention text and DoLa-style re-decode |
| `jspace_viz/verify.py` | Gemini judge and token glosses |
| `jspace_viz/model.py` | model loading, int8 weight-only linear layers |
| `jspace_viz/presets.py` | model/lens presets and the example questions |
| `jspace_viz/static/` | the UI: plain HTML, CSS, JS, no build step |
| `scripts/` | model conversion, lens fitting, evaluations |
| `data/` | evaluation results and calibration tables |
| `colab/` | notebooks to fit the lens on a free GPU |
| `models/`, `lenses/` | downloaded weights and lens files (git-ignored) |

## Setup

**What do I need?**
A Mac with Apple Silicon (8 GB RAM is enough) or a Linux machine with a GPU,
Python 3.12 via `uv`, and about 8 GB of free disk.

**How do I install it?**
```sh
git clone https://github.com/lostfilesofanni/haluviz && cd haluviz
uv venv --python 3.12 && uv pip install -e '.[fit]'
.venv/bin/python scripts/convert_gguf_to_hf.py
```
The conversion downloads a lossless f16 GGUF of the model and rewrites it as a
standard Qwen2 checkpoint in `models/qwen-1_8b-chat-hf`. It takes about five
minutes and ends with a self-check that must print
`self-check answer: The capital of France is Paris.`

**Why convert from GGUF instead of downloading from Hugging Face?**
The original Qwen-1 chat weights are no longer downloadable, and their 2023
modeling code does not run on current transformers. Qwen-1 has the same
tensor layout as Qwen2, which transformers supports natively, so a rename is
enough. The result has no remote code and runs on MPS, CUDA, or CPU.

**Do I need the lens file?**
No. Without `lenses/qwen-1_8b-chat_jlens_wikitext.pt` the app runs in
logit-lens mode. Download the file from the Releases page or fit your own
with `colab/fit_lens_colab.ipynb` (about 45 minutes on a free T4).

**Do I need a Gemini key?**
Only for the Verify button, the Suggest-a-fix "verified correction", and the
English glosses of Chinese tokens. Put `GEMINI_API_KEY=...` in a `.env` file
in the project folder. It is git-ignored. Without a key those features return
"unsure" and everything else works.

## Running it

**How do I start it?**
```sh
.venv/bin/haluviz
```
Then open <http://127.0.0.1:8321>. To open it straight into Chrome from a
terminal on macOS:
```sh
open -a "Google Chrome" http://127.0.0.1:8321
```

**What flags does the server take?**

| flag | default | meaning |
|---|---|---|
| `--preset` | `qwen-1.8b-chat` | model + lens + dtype bundle from `presets.py` |
| `--model-id` | from preset | override the model |
| `--lens` | from preset | path to a fitted lens `.pt` |
| `--dtype` | from preset (`int8`) | `float32`, `float16`, `bfloat16`, `int8` |
| `--host` / `--port` | `127.0.0.1` / `8321` | where to listen |
| `--idle-timeout` | `10` | minutes without a read before exit; `0` = never |

**The page stopped responding after a while. Why?**
The server exits on its own after 10 minutes without a read, so a forgotten
process does not hold 2.4 GB of memory. Start it again, or pass
`--idle-timeout 0`.

**It says the port is already in use.**
Another haluviz is running. The server refuses to start rather than load a
second copy of the model. Stop the old one:
```sh
pkill -f 'haluviz'
```

**How long does a read take?**
About 30 seconds for the first read of a question on an M1 with 32 new
tokens. Set *new tokens* to 16 and it roughly halves. Re-reading the same
question (switching lens, pinning a token, moving the slider) takes a few
seconds because the generated answer is cached.

**How much memory does it use?**
About 2.4 GB while running. Weights are stored as int8 with per-row scales
and dequantized on each call; fp16 weights paged so badly on 8 GB that a
single forward pass took 25 seconds.

## How the server works

**What happens at startup?**
It parses the flags, probes the port and quits if something is listening,
resolves the preset, loads the lens (local file, hub download, or an empty
lens for logit-only mode), loads the model, and then starts uvicorn. If an
idle timeout is set it also starts a watchdog thread.

**What endpoints does it expose?**

| endpoint | what it does |
|---|---|
| `GET /` and `/static/*` | the UI |
| `GET /api/info` | model, lens, device, examples, calibration tables |
| `POST /api/read` | run a prompt, return the layer grid and hallucination signals |
| `POST /api/fix` | abstention text and a DoLa-style re-decode |
| `POST /api/verify` | ask Gemini whether the answer is right |
| `POST /api/gloss` | English glosses for non-English tokens |

**Why is there a lock around the model?**
FastAPI runs sync handlers in a thread pool. Two threads driving the Metal
GPU at once crash the process, so all model use is serialized through one
lock. Requests queue; they do not fail.

**What is cached?**
Greedy generation is deterministic, so the generated token ids are cached per
(prompt, chat flag, max new tokens), up to 64 entries. A re-read skips
generation, which is about two thirds of the time. Token glosses are cached
for the life of the process.

**Can I point it at a different model?**
`presets.py` also lists gpt2, pythia-70m, gemma-3-1b-it, deepseek-coder-1.3b,
llama-3.1-8b-it and deepseek-r1-distill-llama-8b, with lenses pulled from
Neuronpedia or Festyve's releases. Only the Qwen preset has been evaluated and
calibrated; the hallucination percentage is only meaningful there.

## What is on the screen

**The grid.** One column per word, one row per layer. Each cell shows the
word that layer was leaning toward; brighter blue is more confident. Green
columns are words the model generated, orange marks a flagged word, and a
thin orange line on top of a cell means the top choice changed at that layer.

**J-lens vs logit lens.** The logit lens pushes a middle layer's vector
straight through the output matrix, which gives mostly noise (Chinese noise,
for a bilingual model). The J-lens first multiplies by a fitted matrix that
approximates what the rest of the network would do to that vector, so the
middle layers read as real concepts.

**Hallucination check.** A gauge and a percentage: the estimated chance the
answer is wrong, plus the answer text with suspicious words marked. Green is
under 25 %, yellow under 60 %, red above.

**Signals.** For each flagged word, four bars: confidence, how deep the
decision was made, whether the word was in the workspace, and how much the
prediction wavered.

**In the workspace.** The concepts the middle layers were holding while
answering. Chinese tokens get an English gloss when a Gemini key is set.

**Fix.** *Verify answer* asks Gemini. *Suggest a fix* shows an abstention,
a re-decoded answer, and the judge's correction side by side.

**Chat toggle.** On, the prompt is wrapped in the model's chat template, so
it answers like an assistant. Off, it continues the raw text.

**Pinning.** Pin a token to see its rank at every layer and position.

## How the score works

**What is measured per word?**
For every generated word the detector looks at the grid column that produced
it and reads three numbers:

| signal | question it answers | range |
|---|---|---|
| `p_final` | how confident was the output layer in the word it chose | 0 to 1 |
| `commit_depth` | how deep into the network did the word become and stay the top choice | 0 = from the first layer, 1 = only at the output |
| `support` | highest probability the word had anywhere in the middle band of layers (25 % to 85 % of depth) before being said | 0 to 1 |

A fourth, `instability`, is the average Jensen-Shannon divergence between
adjacent layers over the last third of the network, plus the number of top-1
flips there. It is displayed but not yet part of the score.

**How are they combined?**
```
score = mean(1 - p_final, commit_depth, 1 - support)
```
High means the model said something it was unsure of, decided late, and never
had in mind.

**When is a word flagged?**
When its score is at or above the threshold (default 0.80, the slider changes
it), it contains a letter or digit, and it was not copied from the prompt.
Flags apply to whole words: if one sub-word piece of "Nha Trang" crosses the
line, the whole name is marked. Flagged words that touch form a span.

**How is the whole answer scored?**
```
answer_score = (max + mean) / 2    over content-word scores
```
Max alone catches a single wrong name in a correct sentence. Mean alone
catches a fully invented answer. The average was the most robust across both
test sets.

**Where does the percentage come from?**
The answer score is looked up in a calibration table built from 111 real
answers whose truth Gemini judged (100 PopQA questions plus a 30-question hand
set, minus honest refusals). The gauge shows the share of answers in that bin
that were wrong. Which table is used depends on the active lens.

| answer score | wrong, logit lens | wrong, J-lens |
|---|---|---|
| under 0.70 | 8 % | 17 % |
| 0.70 to 0.75 | 36 % | 54 % |
| 0.75 to 0.80 | 75 % | 80 % |
| 0.80 to 0.85 | 91 % | 94 % |
| 0.85 and above | 100 % | 96 to 100 % |

The server re-reads the tables on every `/api/info` call, so re-running the
evaluation updates the UI without a restart.

**Why 0.80 as the default threshold?**
It gave full recall on the 30-question hand set in logit-lens mode. Precision
was 0.56 at that setting, which is why the slider exists.

**Why do the signals look odd under the logit lens?**
Early layers almost never agree with the output under the logit lens, so
commit depth is near 1 and support near 0 for nearly every word. The score
floors around 0.65 and confidence does the separating.

## How good is it

**Headline numbers.**

| | |
|---|---|
| answers evaluated | 111 (82 wrong) |
| AUROC, logit lens | 0.90 |
| AUROC, J-lens | 0.87 |
| model accuracy on PopQA long-tail facts | 12 % |

AUROC 0.90 means a wrong answer scores above a right one 90 % of the time.

**Does the J-lens catch more hallucinations?**
No. It makes the middle layers readable and makes the per-word signals behave
sensibly inside an answer, but at the question level the workspace signals
stay near chance under both lenses and output confidence carries the
detector. The likely reason is the model: it does not know 88 % of long-tail
facts, so "the answer was never in the workspace" is true of right and wrong
answers alike. The J-lens explains how a hallucination formed; it does not
find more of them.

**Does Suggest a fix repair answers?**
The DoLa-style re-decode changes wording but cannot restore knowledge the
model never had. Abstaining (shown when the calibrated risk is 60 % or more)
and the verified correction do work.

**How do I rerun the evaluation?**
Start the app, then:
```sh
.venv/bin/python scripts/eval_detector.py --url http://127.0.0.1:8321          # 30 hand-picked questions
.venv/bin/python scripts/eval_popqa.py --url http://127.0.0.1:8321 --n 100     # 100 PopQA questions
.venv/bin/python scripts/judge_labels.py                                       # relabel answers with Gemini
.venv/bin/python scripts/eval_fix.py --url http://127.0.0.1:8321               # greedy vs DoLa
```
Results land in `data/`.

## Limits

- **Confident errors are invisible.** "Pompeii was buried under Augustus"
  (correct: Titus) was said with no hesitation and scored as uncertain. From
  the inside a confident false belief looks like knowledge. That is what the
  Verify button is for.
- **The model is tiny and old.** It is wrong on most obscure facts and cannot
  count letters, because it sees word pieces.
- **The lens was fitted on 30 texts**, not the paper's 1000, so mid-layer
  readouts are still noisy in the first dozen layers.
- **111 labeled answers is few.** The calibration bins are coarse and the
  equal-weight score was not tuned further on purpose.
- **Gemini is the judge** and it is not perfect either.
- **The J-lens reads single-token concepts.** Multi-token names show as their
  first piece.
- **Speed is bounded by the machine**, not the code.

## Development

**Are there tests?**
Three modules carry a runnable self-check under `if __name__ == "__main__"`:
```sh
.venv/bin/python -m jspace_viz.hallucination   # prints "ok"
.venv/bin/python -m jspace_viz.verify
.venv/bin/python -m jspace_viz.server --help
```

**How do I fit my own lens?**
Locally with `scripts/fit_lens.py --preset qwen-1.8b-chat` if you have the
memory (fitting doubles memory during backward; the 8 GB M1 could not), or on
Colab/Kaggle with the notebooks in `colab/`. Drop the `.pt` into `lenses/`.

**What is planned next?**
Measuring the instability signal and folding it into the score, and a
steering phase: add the lens direction of the correct concept into the middle
layers and regenerate, to test whether a hallucination was a retrieval
failure the workspace can repair.
