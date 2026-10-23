"""Re-label eval rows with the Gemini judge (substring matching is noisy:
"Sangrur is a district in Himachal Pradesh" matches the alias "Sangrur" but is
wrong). Adds a "judge" field to each row in place and reports agreement.

    python scripts/judge_labels.py data/popqa_results.json [data/eval_results.json ...]
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace_viz.verify import judge  # noqa: E402


def main(paths: list[str]) -> None:
    for path in paths:
        doc = json.load(open(path, encoding="utf-8"))
        rows = doc["rows"]
        agree = n = 0
        for i, r in enumerate(rows):
            if r.get("abstained") or not r.get("answer"):
                continue
            if "judge" not in r or r["judge"].get("verdict") == "unsure" and "error" in r["judge"].get("explanation", ""):
                r["judge"] = judge(r["q"], r["answer"])
                json.dump(doc, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            v = r["judge"]["verdict"]
            if v in ("correct", "wrong"):
                n += 1
                agree += (v == "correct") == r["correct"]
            print(f"[{i+1}/{len(rows)}] substring={'✓' if r['correct'] else '✗'} judge={v:7} {r['q'][:40]:<40} → {r['answer'][:45]!r}")
        print(f"\n{path}: judge agreed with substring labels on {agree}/{n} decided rows")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/popqa_results.json"])
