"""Calibrate the hallucination detector on data/questions.json.

Each question is run through read_grid (chat mode); the answer is "correct" if
any gold string appears in the continuation. We then ask: does the detector's
max token score separate wrong answers from right ones?

    python scripts/eval_detector.py [--preset qwen-1.8b-chat] [--mode logit|jlens]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace_viz.analysis import read_grid  # noqa: E402
from jspace_viz.server import load, state  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC: P(score of a wrong answer > score of a right one)."""
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="qwen-1.8b-chat")
    ap.add_argument("--mode", default=None, help="logit | jlens (default: jlens if a lens is loaded)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--questions", default=os.path.join(ROOT, "data", "questions.json"))
    args = ap.parse_args()

    load(args.preset, None, None, None)
    mode = args.mode or ("jlens" if state["lens"].source_layers else "logit")
    questions = json.load(open(args.questions, encoding="utf-8"))

    rows = []
    for q in questions:
        r = read_grid(state["model"], state["lens"], q["q"], mode=mode, chat=True,
                      max_new_tokens=args.max_new_tokens, threshold=args.threshold, top_k=1)
        ans = r["continuation"] or ""
        correct = any(a.lower() in ans.lower() for a in q["answers"])
        h = r["hallucination"]
        rows.append({
            "q": q["q"], "tier": q.get("tier", ""), "answer": ans.strip(), "correct": correct,
            "max_score": h["max_score"], "spans": [s["text"] for s in h["spans"]],
        })
        mark = "✓" if correct else "✗"
        print(f"{mark} {h['max_score']:.2f}  {q['q'][:48]:<48} → {ans.strip()[:70]!r}  flagged={rows[-1]['spans']}")

    scores = np.array([r["max_score"] for r in rows])
    wrong = np.array([not r["correct"] for r in rows])
    flagged = scores >= args.threshold
    tp = int((flagged & wrong).sum())
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int(wrong.sum()), 1)
    # best threshold by F1 over the observed scores
    best = max(
        ((t, 2 * p * rc / max(p + rc, 1e-9)) for t in np.unique(scores)
         for f in [scores >= t] for p in [((f & wrong).sum() / max(f.sum(), 1))] for rc in [((f & wrong).sum() / max(wrong.sum(), 1))]),
        key=lambda x: x[1],
    )
    summary = {
        "mode": mode, "n": len(rows), "n_wrong": int(wrong.sum()), "threshold": args.threshold,
        "auroc": round(auroc(scores, wrong), 3), "precision": round(precision, 3), "recall": round(recall, 3),
        "best_threshold": round(float(best[0]), 3), "best_f1": round(float(best[1]), 3),
    }
    print("\n" + json.dumps(summary, indent=2))
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(ROOT, "data", "eval_results.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
