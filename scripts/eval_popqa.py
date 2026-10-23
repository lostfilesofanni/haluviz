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
    """Empirical P(wrong | score in bin). Monotone-smoothed so the UI never
    shows a lower risk for a higher score."""
    s, w = np.asarray(scores), np.asarray(wrong, bool)
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (s >= lo) & (s < hi)
        bins.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "p_wrong": float(w[m].mean()) if m.any() else None})
    last = 0.0
    for b in bins:  # fill empty bins and enforce monotonicity
        if b["p_wrong"] is None or b["p_wrong"] < last:
            b["p_wrong"] = last
        last = b["p_wrong"]
    return bins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8321")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--resume", action="store_true", help="continue a previous run from data/popqa_results.json")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]

    info = json.load(urllib.request.urlopen(args.url.rstrip("/") + "/api/info"))
    mode = args.mode or ("jlens" if info["fitted_layers"] else "logit")
    out_path = os.path.join(ROOT, "data", "popqa_results.json")
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
        h = r["hallucination"]
        ct = content_tokens(h["tokens"])
        rows.append({
            "id": ex["id"], "q": ex["question"], "gold": aliases[:3], "answer": ans, "correct": correct,
            "max_score": max((t["score"] for t in ct), default=0.0),
            "mean_score": float(np.mean([t["score"] for t in ct])) if ct else 0.0,
            "min_p_final": min((t["p_final"] for t in ct), default=1.0),
            "tokens": h["tokens"], "mode": mode,
        })
        print(f"[{k+1}/{len(idx)}] {'✓' if correct else '✗'} {rows[-1]['max_score']:.2f} {ex['question'][:50]:<50} → {ans[:60]!r}  ({(time.time()-t0)/(k+1):.0f}s/q)")
        json.dump({"rows": rows}, open(out_path, "w"), indent=1, ensure_ascii=False)

    wrong = [not r["correct"] for r in rows]
    scores = [r["max_score"] for r in rows]
    summary = {
        "mode": mode, "n": len(rows), "n_wrong": int(sum(wrong)), "accuracy": round(1 - sum(wrong) / len(rows), 3),
        "auroc_max_score": round(auroc(scores, wrong), 3),
        "auroc_min_p_final": round(auroc([1 - r["min_p_final"] for r in rows], wrong), 3),
        "auroc_mean_score": round(auroc([r["mean_score"] for r in rows], wrong), 3),
    }
    bins = calibrate(scores, wrong)
    print("\n" + json.dumps(summary, indent=2))
    for b in bins:
        print(f"  score {b['lo']:.2f}–{b['hi']:.2f}: n={b['n']:3d}  P(wrong)={b['p_wrong']:.2f}")
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1, ensure_ascii=False)
    json.dump({"source": "PopQA", "mode": mode, "n": len(rows), "metric": "max_score", "bins": bins, "summary": summary},
              open(os.path.join(ROOT, "data", "calibration.json"), "w"), indent=2)
    print("wrote data/calibration.json")


if __name__ == "__main__":
    main()
