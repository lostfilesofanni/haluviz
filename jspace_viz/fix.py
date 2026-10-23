"""Phase 2: suggest a fix for a flagged answer.

DoLa-style contrastive decoding (Chuang et al. 2023, "Decoding by Contrasting
Layers"): at each step, pick a *premature* layer whose next-token distribution
differs most from the final layer's (max Jensen–Shannon divergence), and score
tokens by  log p_final − log p_premature  over the plausible set
(p_final ≥ α · max p_final). Facts that *emerge* through the network get
boosted; tokens the early layers already liked (surface patterns, frequent
words) get suppressed. We already have every layer's residual, so this is
cheap.

# ponytail: greedy only, fixed candidate bucket (every 2nd layer in the lower
# 2/3), α = 0.1 as in the paper. No repetition penalty.
"""

from __future__ import annotations

import torch

from jspace_viz.model import WrappedModel


def _jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
                  + (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1))


@torch.no_grad()
def dola_generate(model: WrappedModel, input_ids: torch.Tensor, *, max_new_tokens: int = 32,
                  alpha: float = 0.1, candidates: list[int] | None = None) -> dict:
    n = model.n_layers
    candidates = candidates or list(range(2, (2 * n) // 3, 2))
    eos = {model.tokenizer.eos_token_id, 151643, 151645}
    ids = input_ids
    past = None
    chosen_layers: list[int] = []
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        res = model.hf_model(input_ids=ids if past is None else ids[:, -1:], past_key_values=past,
                             use_cache=True, output_hidden_states=True)
        past = res.past_key_values
        hs = res.hidden_states  # tuple: embeddings + one per layer; hs[l+1] = output of layer l
        final = res.logits[0, -1].float().log_softmax(-1)
        p_final = final.exp()
        # premature distributions from the candidate layers (same norm + head as the output)
        prem = torch.stack([model.unembed(hs[l + 1][0, -1].float()).float().log_softmax(-1) for l in candidates])
        js = _jsd(p_final.unsqueeze(0), prem.exp())
        k = int(js.argmax())
        chosen_layers.append(candidates[k])
        plausible = p_final >= alpha * p_final.max()
        contrast = torch.where(plausible, final - prem[k], torch.full_like(final, -float("inf")))
        nxt = int(contrast.argmax())
        out_ids.append(nxt)
        if nxt in eos:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    text = model.tokenizer.decode(out_ids, skip_special_tokens=True)
    return {"text": text, "ids": out_ids, "premature_layers": chosen_layers}


def abstain_text(p_wrong: float | None, score: float) -> str | None:
    """The safer answer when the risk is high: say so instead of guessing."""
    risk = p_wrong if p_wrong is not None else score
    if risk < 0.6:
        return None
    return "I'm not confident about this. I might be confusing it with something else, so please verify it elsewhere."
