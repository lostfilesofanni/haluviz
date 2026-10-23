"""Evaluate and calibrate the detector on PopQA (long-tail facts, auto-labeled).

PopQA (akariasai/PopQA) ships gold answers plus accepted aliases, so no hand
labeling: the answer is "correct" if any alias appears in the continuation.

    python scripts/eval_popqa.py --url http://127.0.0.1:8321 --n 300

Writes data/popqa_results.json (per-question rows with per-token signals) and
data/calibration.json: score bins -> empirical P(wrong), which the UI uses to
show "estimated chance this answer is wrong" instead of a bare threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
WORD = re.compile(r"[A-Za-z0-9]")
BINS = [0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
# "I don't know" is honest, not a hallucination: kept out of the calibration.
ABSTAIN = re.compile(r"i('m| am) sorry|i (do not|don't|cannot|can't) (have|know|find|provide|recall)|no information|not (sure|aware|familiar)", re.I)


def content_tokens(tokens):
    out = []
    for t in tokens:
        if t["token"].startswith("<|"):
            break
        if WORD.search(t["token"]) and not t["in_prompt"]:
            out.append(t)
    return out


def auroc(scores, positive):
    s, p = np.asarray(scores), np.asarray(positive, bool)
    a, b = s[p], s[~p]
    if not len(a) or not len(b):
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def calibrate(scores, wrong):
    """Empirical P(wrong | score bin), made monotone by pool-adjacent-violators
    (weighted isotonic regression) so a higher score never shows lower risk."""
    s, w = np.asarray(scores), np.asarray(wrong, bool)
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (s >= lo) & (s < hi)
        bins.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "sum": int(w[m].sum())})
    blocks = [[b] for b in bins if b["n"]]
    i = 0
    while i < len(blocks) - 1:
        a, b = blocks[i], blocks[i + 1]
        pa = sum(x["sum"] for x in a) / sum(x["n"] for x in a)
        pb = sum(x["sum"] for x in b) / sum(x["n"] for x in b)
        if pa > pb:
            blocks[i:i + 2] = [a + b]
            i = max(i - 1, 0)
        else:
            i += 1
    for blk in blocks:
        p = sum(x["sum"] for x in blk) / sum(x["n"] for x in blk)
        for x in blk:
            x["p_wrong"] = round(p, 3)
    last = 0.0
    for b in bins:  # empty bins inherit the previous block's risk
        b["p_wrong"] = b.get("p_wrong", last)
        last = b["p_wrong"]
        b.pop("sum", None)
    return bins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8321")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--resume", action="store_true", help="continue a previous run from data/popqa_results.json")
    ap.add_argument("--extra", nargs="*", default=[], help="other results files (e.g. data/eval_results.json) pooled into the calibration")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _jlens (calibration.json is only written untagged)")
    ap.add_argument("--labels-from", nargs="*", default=[], help="results files whose judge labels are reused when question+answer match")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]

    info = json.load(urllib.request.urlopen(args.url.rstrip("/") + "/api/info"))
    mode = args.mode or ("jlens" if info["fitted_layers"] else "logit")
    out_path = os.path.join(ROOT, "data", f"popqa_results{args.tag}.json")
    rows = []
    if args.resume and os.path.exists(out_path):
        rows = json.load(open(out_path))["rows"]
    done = {r["id"] for r in rows}

    def run(q):
        body = json.dumps({"prompt": q, "mode": mode, "chat": True, "top_k": 1,
                           "max_new_tokens": args.max_new_tokens}).encode()
        req = urllib.request.Request(args.url.rstrip("/") + "/api/read", body, {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=900))

    t0 = time.time()
    for k, i in enumerate(idx):
        ex = ds[i]
        if ex["id"] in done:
            continue
        aliases = json.loads(ex["possible_answers"])
        r = run(ex["question"])
        ans = (r["continuation"] or "").strip()
        correct = any(a.lower() in ans.lower() for a in aliases if a.strip())
        abstained = bool(ABSTAIN.search(ans)) and not correct
        h = r["hallucination"]
        ct = content_tokens(h["tokens"])
        rows.append({
            "id": ex["id"], "q": ex["question"], "gold": aliases[:3], "answer": ans, "correct": correct, "abstained": abstained,
            "max_score": max((t["score"] for t in ct), default=0.0),
            "mean_score": float(np.mean([t["score"] for t in ct])) if ct else 0.0,
            "min_p_final": min((t["p_final"] for t in ct), default=1.0),
            "tokens": h["tokens"], "mode": mode,
        })
        print(f"[{k+1}/{len(idx)}] {'✓' if correct else '✗'} {rows[-1]['max_score']:.2f} {ex['question'][:50]:<50} → {ans[:60]!r}  ({(time.time()-t0)/(k+1):.0f}s/q)")
        json.dump({"rows": rows}, open(out_path, "w"), indent=1, ensure_ascii=False)

    pooled = list(rows)
    for path in args.extra:  # pool other labeled runs so the calibration isn't one dataset's base rate
        for r in json.load(open(path, encoding="utf-8"))["rows"]:
            r.setdefault("abstained", False)
            pooled.append(r)
    judged = {}
    for path in args.labels_from:  # greedy output is identical across lens modes, so labels transfer
        for r in json.load(open(path, encoding="utf-8"))["rows"]:
            if "judge" in r:
                judged[(r["q"], r["answer"])] = r["judge"]
    for r in pooled:
        if "judge" not in r and (r["q"], r["answer"]) in judged:
            r["judge"] = judged[(r["q"], r["answer"])]
    for r in pooled:  # rows from an older run lack the field
        r.setdefault("abstained", bool(ABSTAIN.search(r["answer"])) and not r["correct"])
        if r.get("judge", {}).get("verdict") in ("correct", "wrong"):  # scripts/judge_labels.py
            r["correct"] = r["judge"]["verdict"] == "correct"
        ct = content_tokens(r["tokens"])
        r["answer_score"] = round((max(t["score"] for t in ct) + np.mean([t["score"] for t in ct])) / 2, 3) if ct else 0.0
        r["max_score"] = max((t["score"] for t in ct), default=0.0)
        r["mean_score"] = float(np.mean([t["score"] for t in ct])) if ct else 0.0
        r["min_p_final"] = min((t["p_final"] for t in ct), default=1.0)
    n_abstain = sum(r["abstained"] for r in pooled)
    rows_c = [r for r in pooled if not r["abstained"]]
    wrong = [not r["correct"] for r in rows_c]
    scores = [r["answer_score"] for r in rows_c]
    summary = {
        "mode": mode, "n": len(pooled), "sources": ["PopQA"] + args.extra, "n_abstained": int(n_abstain), "n_scored": len(rows_c),
        "n_wrong": int(sum(wrong)), "accuracy": round(1 - sum(wrong) / max(len(rows_c), 1), 3),
        "labels": "judge" if any("judge" in r for r in rows) else "substring",
        "auroc_answer_score": round(auroc(scores, wrong), 3),
        "auroc_max_score": round(auroc([r["max_score"] for r in rows_c], wrong), 3),
        "auroc_min_p_final": round(auroc([1 - r["min_p_final"] for r in rows_c], wrong), 3),
        "auroc_mean_score": round(auroc([r["mean_score"] for r in rows_c], wrong), 3),
    }
    bins = calibrate(scores, wrong)
    print("\n" + json.dumps(summary, indent=2))
    for b in bins:
        print(f"  score {b['lo']:.2f}–{b['hi']:.2f}: n={b['n']:3d}  P(wrong)={b['p_wrong']:.2f}")
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1, ensure_ascii=False)
    cal_path = os.path.join(ROOT, "data", f"calibration{args.tag}.json")
    json.dump({"source": "PopQA" + (" + hand set" if args.extra else ""), "mode": mode, "n": len(rows_c), "metric": "answer_score", "bins": bins, "summary": summary},
              open(cal_path, "w"), indent=2)
    print("wrote", cal_path)


if __name__ == "__main__":
    main()
