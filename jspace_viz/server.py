# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""FastAPI server: loads one model + lens, serves the live visualizer."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jspace_viz.analysis import read_grid
from jspace_viz.fix import abstain_text, dola_generate
from jspace_viz.lens import JacobianLens
from jspace_viz.model import load_model
from jspace_viz.presets import EXAMPLES, PRESETS
from jspace_viz.verify import api_key, gloss, judge

logger = logging.getLogger("jspace_viz")
STATIC_DIR = Path(__file__).parent / "static"

state: dict[str, Any] = {}
# MPS is not safe to drive from two threads at once (Metal command-buffer
# assertion); FastAPI runs sync endpoints in a pool, so serialize model use.
_model_lock = threading.Lock()


def _idle_watchdog(idle_seconds: float) -> None:
    """Exit when nobody has asked for a read in a while — the model holds GBs
    of wired memory, and a forgotten server starves an 8 GB machine."""
    while True:
        time.sleep(30)
        if time.monotonic() - state.get("last_read", time.monotonic()) > idle_seconds:
            logger.info("idle for %.0f min; exiting to free memory", idle_seconds / 60)
            os._exit(0)


class ReadRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    mode: str = "jlens"  # "jlens" | "logit"
    top_k: int = Field(default=8, ge=1, le=50)
    max_seq_len: int = Field(default=192, ge=8, le=512)
    chat: bool = False
    pinned_ids: list[int] = Field(default_factory=list, max_length=16)
    continuation: bool = True  # False = latency-sensitive live-typing read
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)


app = FastAPI(title="jspace-viz")


@app.get("/api/info")
def info() -> dict[str, Any]:
    model, lens = state["model"], state["lens"]
    return {
        "model_id": state["model_id"],
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "fitted_layers": lens.source_layers,
        "lens_n_prompts": lens.n_prompts,
        "lens_source": state["lens_source"],
        "examples": EXAMPLES,
        "device": str(model.device),
        "chip_baseline": state.get("chip_baseline", {}),
        "verifier": api_key() is not None,
        "calibration": load_calibration(),
        "calibration_jlens": load_calibration("_jlens"),
    }


def load_calibration(tag: str = "") -> dict[str, Any] | None:
    """Score-bin -> P(wrong) table from scripts/eval_popqa.py, if it has been run."""
    path = Path(__file__).parent.parent / "data" / f"calibration{tag}.json"
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return None


class FixRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    chat: bool = True
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    p_wrong: float | None = None
    score: float = 0.0


@app.post("/api/fix")
def fix(req: FixRequest) -> dict[str, Any]:
    """Phase 2: abstention text + DoLa-style contrastive re-decode."""
    model = state["model"]
    prompt = req.prompt
    if req.chat:
        prompt = model.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    state["last_read"] = time.monotonic()
    try:
        with _model_lock:
            ids = model.encode(prompt, max_length=512)
            dola = dola_generate(model, ids, max_new_tokens=req.max_new_tokens)
    except Exception as exc:
        logger.exception("fix failed")
        raise HTTPException(500, str(exc)) from exc
    return {"abstain": abstain_text(req.p_wrong, req.score), "dola": dola["text"].strip(), "premature_layers": dola["premature_layers"]}


class GlossRequest(BaseModel):
    tokens: list[str] = Field(max_length=80)


_gloss_cache: dict[str, str] = {}


@app.post("/api/gloss")
def gloss_tokens(req: GlossRequest) -> dict[str, str]:
    """English glosses for non-English tokens, cached for the process."""
    missing = [t for t in dict.fromkeys(req.tokens) if t not in _gloss_cache]
    if missing:
        _gloss_cache.update(gloss(missing))
    return {t: _gloss_cache[t] for t in req.tokens if t in _gloss_cache}


class VerifyRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=4000)


@app.post("/api/verify")
def verify(req: VerifyRequest) -> dict[str, Any]:
    """Ask the external judge whether the answer is factually right."""
    return judge(req.question, req.answer)


def load_chip_baseline(model_id: str) -> dict[str, float]:
    """Neutral-text calibration for the workspace panel (see
    ``scripts/make_chip_baseline.py``); empty when none exists for the model."""
    path = Path(__file__).parent / "data" / "chip_baselines" / f"{model_id.split('/')[-1]}.json"
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@app.post("/api/read")
def read(req: ReadRequest) -> dict[str, Any]:
    if req.mode not in ("jlens", "logit"):
        raise HTTPException(400, "mode must be 'jlens' or 'logit'")
    state["last_read"] = time.monotonic()
    try:
        with _model_lock:
            return read_grid(
                state["model"],
                state["lens"],
                req.prompt,
                mode=req.mode,
                top_k=req.top_k,
                max_seq_len=req.max_seq_len,
                pinned_ids=req.pinned_ids,
                chat=req.chat,
                generate_continuation=req.continuation,
                max_new_tokens=req.max_new_tokens,
                threshold=req.threshold,
            )
    except Exception as exc:  # surfaced to the UI status line
        logger.exception("read failed")
        raise HTTPException(500, str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def load(preset_name: str | None, model_id: str | None, lens_path: str | None, dtype: str | None) -> None:
    """Resolve preset/overrides, load model + lens into module state."""
    lens: JacobianLens | None = None
    if preset_name:
        if preset_name not in PRESETS:
            raise SystemExit(f"unknown preset {preset_name!r}; options: {sorted(PRESETS)}")
        preset = PRESETS[preset_name]
        model_id = model_id or preset.model_id
        dtype = dtype or preset.dtype
        if lens_path is None:
            state["lens_source"] = ":".join(preset.lens)
            kind, *rest = preset.lens
            if kind == "hub":
                repo, filename = rest
                logger.info("downloading lens %s :: %s", repo, filename)
                lens = JacobianLens.from_pretrained(repo, filename=filename)
            else:
                (path,) = rest
                if not os.path.isabs(path) and not os.path.exists(path):
                    # Relative preset paths resolve against the repo root, so
                    # the server works regardless of the launch directory.
                    repo_root = Path(__file__).parent.parent
                    path = str(repo_root / path)
                if not os.path.exists(path) and preset.lens_fallback is not None:
                    repo, filename = preset.lens_fallback
                    logger.info("no local lens; downloading %s :: %s", repo, filename)
                    lens = JacobianLens.from_pretrained(repo, filename=filename)
                    state["lens_source"] = f"hub:{repo}:{filename}"
                elif not os.path.exists(path):
                    # No lens yet: run logit-lens only. Fit one with
                    #   python scripts/fit_lens.py --preset <name>
                    logger.warning("lens file %r not found; logit-lens mode only", path)
                    lens = JacobianLens({}, n_prompts=0, d_model=0)
                    state["lens_source"] = "none (logit-lens only)"
                else:
                    lens = JacobianLens.load(path)
    if lens_path is not None:
        lens = JacobianLens.load(lens_path)
        state["lens_source"] = lens_path
    if model_id is None or lens is None:
        raise SystemExit("pass --preset, or both --model-id and --lens")

    logger.info("loading model %s", model_id)
    model = load_model(model_id, dtype=dtype or "auto")
    if lens.d_model == 0:
        lens.d_model = model.d_model
    if lens.d_model != model.d_model:
        raise SystemExit(
            f"lens d_model={lens.d_model} does not match model d_model={model.d_model}"
        )
    state.update(
        model=model,
        lens=lens,
        model_id=model_id,
        chip_baseline=load_chip_baseline(model_id),
    )
    logger.info("ready: %s + %r on %s", model_id, lens, model.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live J-space visualizer")
    parser.add_argument(
        "--preset", default=os.environ.get("JSPACE_PRESET", "qwen-1.8b-chat")
    )
    parser.add_argument("--model-id", default=None, help="override the preset's model")
    parser.add_argument("--lens", default=None, help="path to a fitted lens .pt")
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16", "int8"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--idle-timeout", type=float, default=10.0, help="minutes without a read before the server exits (0 = never)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load(args.preset, args.model_id, args.lens, args.dtype)
    state["last_read"] = time.monotonic()
    if args.idle_timeout > 0:
        threading.Thread(target=_idle_watchdog, args=(args.idle_timeout * 60,), daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
