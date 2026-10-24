"""Per-token hallucination signals read off the layer x position grid.

A fourth, *instability*, is reported but not scored until measured: the mean
Jensen–Shannon divergence between adjacent layers' predictions over the last
third of the network (how much the model's mind kept changing), plus the
number of top-1 flips there.

For every *generated* token we ask three questions of the readout column that
produced it (position t-1 predicts token t):

  p_final       how confident was the output distribution in the token it chose
  commit_depth  how deep into the network did that token become (and stay) top-1
                — 0.0 = committed from the first layer, 1.0 = only at the output
  support       max probability the token had anywhere in the mid-depth
                "workspace" band (25–85% of fitted layers) before being emitted

score = mean(1 - p_final, commit_depth, 1 - support). High = the model said
something it had low confidence in, decided late, and never "had in mind".

Default threshold 0.8 was set from scripts/eval_detector.py on data/questions.json
(logit-lens mode): under the logit lens commit_depth≈1 and support≈0 for nearly
every token, so the score floors near 0.65 and only p_final moves it.
# ponytail: equal weights + one global threshold; re-tune once a J-lens is fitted.
"""

from __future__ import annotations

import re
from typing import Any

WORDLIKE = re.compile(r"[A-Za-z0-9]")


def hallucination_signals(
    grid: list[dict[str, Any]],
    context_ids: list[int],
    vocab: dict[int, str],
    prompt_len: int,
    *,
    threshold: float = 0.8,
) -> dict[str, Any]:
    fitted = [r for r in grid if not r["is_output"]]
    out = next(r for r in grid if r["is_output"])
    n = len(fitted)
    lo, hi = int(n * 0.25), max(int(n * 0.25) + 1, int(n * 0.85))
    prompt_ids = set(context_ids[:prompt_len])  # copying a prompt word is not hallucinating
    tokens: list[dict[str, Any]] = []
    for t in range(prompt_len, len(context_ids)):
        tid, col = context_ids[t], t - 1
        s = vocab.get(tid, "")
        if s.startswith("<|") and s.endswith("|>"):
            break  # end-of-turn marker: the answer is over
        p_final = out["next_prob"][col]
        commit = n
        for i in range(n - 1, -1, -1):
            if fitted[i]["top_ids"][col][0] != tid:
                break
            commit = i
        support = max((fitted[i]["next_prob"][col] for i in range(lo, hi)), default=0.0)
        tail = grid[-max(1, len(grid) // 3):]  # last third incl. output row
        instability = sum(r.get("jsd_prev", [0] * (col + 1))[col] for r in tail) / len(tail)
        flips = sum(tail[i]["top_ids"][col][0] != tail[i - 1]["top_ids"][col][0] for i in range(1, len(tail)))
        score = ((1 - p_final) + commit / max(n, 1) + (1 - support)) / 3
        tokens.append(
            {
                "pos": t,
                "token": s,
                "p_final": round(p_final, 4),
                "commit_depth": round(commit / max(n, 1), 3),
                "support": round(support, 4),
                "instability": round(instability, 3),  # 4th signal, displayed, not yet in score
                "flips": flips,
                "score": round(score, 3),
                "in_prompt": tid in prompt_ids,
                "flag": bool(score >= threshold and WORDLIKE.search(s) and tid not in prompt_ids),
            }
        )
    # Flags apply to whole words: a sub-word piece over the line ("ha" in
    # " N|ha| Tr|ang") drags its siblings along, so spans read as words.
    word_start = 0
    for i, tok in enumerate(tokens):
        if i > 0 and (tok["token"].startswith(" ") or not WORDLIKE.search(tok["token"])):
            word_start = i
        tok["word"] = word_start
    flagged_words = {tok["word"] for tok in tokens if tok["flag"]}
    for tok in tokens:
        tok["flag"] = tok["word"] in flagged_words and bool(WORDLIKE.search(tok["token"]))
    spans, cur = [], None
    for tok in tokens:
        if tok["flag"]:
            if cur and cur["end"] == tok["pos"] - 1:
                cur["end"], cur["text"], cur["score"] = tok["pos"], cur["text"] + tok["token"], max(cur["score"], tok["score"])
            else:
                cur = {"start": tok["pos"], "end": tok["pos"], "text": tok["token"], "score": tok["score"]}
                spans.append(cur)
        else:
            cur = None
    for sp in spans:
        sp["text"] = sp["text"].strip()
    content = [t["score"] for t in tokens if WORDLIKE.search(t["token"]) and not t["in_prompt"]]
    return {
        "threshold": threshold,
        "tokens": tokens,
        "spans": spans,
        "max_score": max(content, default=0.0),
        # answer-level risk: (max + mean)/2 of content-token scores. Max alone
        # catches a single wrong entity in a correct frame (hand set, AUROC .92);
        # mean alone catches whole fabricated answers (PopQA, .81); the average
        # holds up on both (.90 / .82).
        "answer_score": round((max(content) + sum(content) / len(content)) / 2, 3) if content else 0.0,
    }


if __name__ == "__main__":  # self-check: late-committed, unsupported, low-p token must be flagged
    grid = [
        {"is_output": False, "top_ids": [[1], [1]], "next_prob": [0.9, 0.0]},
        {"is_output": False, "top_ids": [[1], [3]], "next_prob": [0.9, 0.0]},
        {"is_output": False, "top_ids": [[1], [3]], "next_prob": [0.9, 0.0]},
        {"is_output": False, "top_ids": [[1], [3]], "next_prob": [0.9, 0.0]},
        {"is_output": True, "top_ids": [[1], [2]], "next_prob": [0.95, 0.2]},
    ]
    r = hallucination_signals(grid, [0, 1, 2], {1: " Paris", 2: " Zork"}, 1)
    assert hallucination_signals(grid, [2, 1, 2], {1: " Paris", 2: " Zork"}, 1)["spans"] == []  # prompt copy
    assert r["tokens"][0]["flag"] is False and r["tokens"][0]["commit_depth"] == 0.0
    assert r["tokens"][1]["flag"] is True and r["tokens"][1]["commit_depth"] == 1.0
    assert r["spans"] == [{"start": 2, "end": 2, "text": "Zork", "score": r["tokens"][1]["score"]}]
    print("ok")
