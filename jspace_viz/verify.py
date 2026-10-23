"""Verifier: ask a stronger model whether Qwen's answer is actually right.

The detector reads the model's internals and says "it was guessing". This
reads the world and says "it was wrong, here's the answer". Gemini via the
REST API; no SDK dependency. Key from GEMINI_API_KEY or a .env file.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

PROMPT = """You are a strict fact checker. A small language model was asked a question and gave an answer.
Decide whether the answer is factually correct. Reply with JSON only, no prose, in this exact shape:
{{"verdict": "correct" | "wrong" | "unsure", "correct_answer": "<the right answer, short>", "explanation": "<one sentence>"}}
"unsure" is for questions with no single factual answer or that you cannot judge.

Question: {question}
Model's answer: {answer}"""


def api_key() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def judge(question: str, answer: str, *, model: str | None = None, timeout: int = 60) -> dict:
    key = api_key()
    if not key:
        return {"verdict": "unsure", "correct_answer": "", "explanation": "no GEMINI_API_KEY configured (put it in .env)"}
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(question=question, answer=answer)}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        GEMINI_URL.format(model=model, key=key),
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0) if m else text)
        out.setdefault("verdict", "unsure")
        if out["verdict"] not in ("correct", "wrong", "unsure"):
            out["verdict"] = "unsure"
        return {"verdict": out["verdict"], "correct_answer": str(out.get("correct_answer", "")),
                "explanation": str(out.get("explanation", "")), "judge": model}
    except Exception as exc:  # the UI shows this; never crash a read over a verifier hiccup
        return {"verdict": "unsure", "correct_answer": "", "explanation": f"verifier error: {exc}"}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 2 else "What is the capital city of Palau?"
    a = sys.argv[2] if len(sys.argv) > 2 else "The capital city of Palau is Nha Trang."
    print(json.dumps(judge(q, a), indent=2))
