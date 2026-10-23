"""Does the phase-2 re-decode actually fix anything? Greedy vs DoLa on the
hand set, judged by Gemini.

    python scripts/eval_fix.py --url http://127.0.0.1:8321
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace_viz.verify import judge  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8321")
    ap.add_argument("--questions", default=os.path.join(ROOT, "data", "questions.json"))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    qs = json.load(open(args.questions, encoding="utf-8"))
    rows, n_g, n_d = [], 0, 0
    for q in qs:
        g = post(args.url, "/api/read", {"prompt": q["q"], "mode": "logit", "chat": True, "top_k": 1, "max_new_tokens": args.max_new_tokens})
        d = post(args.url, "/api/fix", {"prompt": q["q"], "max_new_tokens": args.max_new_tokens})
        ga, da = (g["continuation"] or "").strip(), d["dola"]
        jg = judge(q["q"], ga)["verdict"]
        jd = jg if da == ga else judge(q["q"], da)["verdict"]
        n_g += jg == "correct"
        n_d += jd == "correct"
        rows.append({"q": q["q"], "greedy": ga, "greedy_verdict": jg, "dola": da, "dola_verdict": jd, "changed": da != ga})
        print(f"greedy {jg:7} dola {jd:7} {'≠' if da != ga else '='}  {q['q'][:45]:<45} → {da[:60]!r}")
    summary = {"n": len(rows), "greedy_correct": n_g, "dola_correct": n_d,
               "changed": sum(r["changed"] for r in rows),
               "fixed": sum(r["greedy_verdict"] != "correct" and r["dola_verdict"] == "correct" for r in rows),
               "broken": sum(r["greedy_verdict"] == "correct" and r["dola_verdict"] != "correct" for r in rows)}
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(ROOT, "data", "fix_results.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
