"""Metrics for quality, latency, visible stability, and commitment behavior."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import PreTrainedTokenizerBase


def clean_bpe_token(token: str) -> str:
    """Remove common tokenizer boundary markers before character counting."""
    return token.replace("▁", "").replace("Ġ", "").replace("##", "").replace(" ", "")


def build_character_mass_tensor(tokenizer: PreTrainedTokenizerBase, device: torch.device) -> torch.Tensor:
    """Return tensor c[id] = visible character mass of a token.

    Special tokens receive zero mass because they are not displayed.
    """
    vocab = tokenizer.get_vocab()
    mass = torch.zeros(len(tokenizer), dtype=torch.float32, device=device)
    special_ids = set(tokenizer.all_special_ids or [])
    for token, token_id in vocab.items():
        if token_id in special_ids:
            mass[token_id] = 0.0
        else:
            mass[token_id] = float(len(clean_bpe_token(token)))
    return mass


def lcp_length(a: str, b: str) -> int:
    """Length of longest common prefix between two strings."""
    upto = min(len(a), len(b))
    for i in range(upto):
        if a[i] != b[i]:
            return i
    return upto


def detokenized_erasure_step(previous: str, current: str) -> int:
    """Character count removed when moving from previous visible string to current."""
    return max(0, len(previous) - lcp_length(previous, current))


def total_detokenized_erasure(visible_stream: Sequence[str]) -> int:
    """Total Detokenized Erasure over a visible output stream."""
    if len(visible_stream) <= 1:
        return 0
    return sum(
        detokenized_erasure_step(visible_stream[i - 1], visible_stream[i])
        for i in range(1, len(visible_stream))
    )


def normalized_detokenized_erasure(visible_stream: Sequence[str], eps: float = 1e-8) -> float:
    """Length-normalized Detokenized Erasure."""
    denom = sum(len(z) for z in visible_stream) + eps
    return float(total_detokenized_erasure(visible_stream) / denom)


def char_mass_from_ids(ids: Sequence[int], char_mass: torch.Tensor) -> float:
    """Sum character mass for token IDs."""
    if not ids:
        return 0.0
    idx = torch.as_tensor(list(ids), dtype=torch.long, device=char_mass.device)
    return float(char_mass[idx].sum().item())


def cnal_step(
    source_ids_read: Sequence[int],
    visible_text: str,
    char_mass: torch.Tensor,
    gamma: float,
) -> Tuple[float, float, float]:
    """Compute step-level CNAL lag and its two components."""
    c_src = char_mass_from_ids(source_ids_read, char_mass)
    c_vis = float(len(visible_text))
    lag = max(0.0, c_src - gamma * c_vis)
    return lag, c_src, c_vis


def commit_delay(read_positions: Sequence[int]) -> float:
    """Mean source read position at which committed tokens first appear."""
    if not read_positions:
        return 0.0
    return float(np.mean(read_positions))


def character_weighted_commit_delay(read_positions: Sequence[int], token_masses: Sequence[float]) -> float:
    """Character-weighted source read position for committed tokens."""
    if not read_positions or not token_masses:
        return 0.0
    weights = np.asarray(token_masses, dtype=float)
    reads = np.asarray(read_positions, dtype=float)
    denom = weights.sum()
    return float((weights * reads).sum() / denom) if denom > 0 else 0.0


def corpus_quality_metrics(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    comet_model: object | None = None,
) -> Dict[str, float]:
    """Compute corpus-level SacreBLEU, chrF++, and optional COMET.

    SacreBLEU expects a list of reference streams. For a single reference set,
    pass [references], not [[r] for r in references].
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    scores = {
        "BLEU": float(BLEU(tokenize="13a", effective_order=True).corpus_score(predictions, [references]).score),
        "chrF++": float(CHRF(word_order=2).corpus_score(predictions, [references]).score),
    }

    if comet_model is not None:
        if sources is None:
            raise ValueError("sources are required for COMET scoring.")
        data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
        output = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        scores["COMET"] = float(getattr(output, "system_score", output.get("system_score")))

    return scores
