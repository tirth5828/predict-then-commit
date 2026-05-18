"""Autoregressive decoding utilities for custom and streaming baselines."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from ptc.model import StreamingTransformer


def decoder_start_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return decoder start token ID, falling back to pad token for Marian-style models."""
    start = getattr(tokenizer, "decoder_start_token_id", None)
    if start is not None:
        return int(start)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id.")
    return int(tokenizer.pad_token_id)


def encode_source_prefix(tokenizer: PreTrainedTokenizerBase, text: str) -> List[int]:
    """Encode source text without special tokens."""
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def add_eos(ids: Sequence[int], tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """Append EOS if the tokenizer defines one."""
    eos = tokenizer.eos_token_id
    if eos is None:
        return list(map(int, ids))
    return list(map(int, ids)) + [int(eos)]


def decode_ids(tokenizer: PreTrainedTokenizerBase, ids: Sequence[int]) -> str:
    """Decode token IDs to a clean string."""
    return tokenizer.decode(list(map(int, ids)), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


@torch.no_grad()
def greedy_decode(
    model: StreamingTransformer,
    tokenizer: PreTrainedTokenizerBase,
    src_ids: Sequence[int],
    max_new_tokens: int,
    block_pad: bool = True,
) -> Tuple[List[int], str, float, torch.Tensor]:
    """True greedy autoregressive decoding.

    Returns generated IDs, decoded text, mean token entropy, and final decoder state.
    """
    model.eval()
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    start_id = decoder_start_id(tokenizer)

    src = torch.tensor([list(src_ids)], dtype=torch.long, device=device)
    tgt = torch.tensor([[start_id]], dtype=torch.long, device=device)

    generated: List[int] = []
    entropies: List[float] = []
    last_state = torch.zeros(1, model.d_model, device=device)

    for _ in range(max_new_tokens):
        mem_mask = torch.zeros((tgt.size(1), src.size(1)), dtype=torch.float32, device=device)
        logits, states = model(src, tgt, memory_mask=mem_mask)
        next_logits = logits[:, -1, :].clone()
        last_state = states[:, -1, :]

        if block_pad:
            next_logits[:, pad_id] = -float("inf")

        probs = torch.softmax(next_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).item()
        entropies.append(float(entropy))

        next_id = int(torch.argmax(next_logits, dim=-1).item())
        if eos_id is not None and next_id == int(eos_id):
            break

        generated.append(next_id)
        tgt = torch.cat([tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    mean_entropy = float(sum(entropies) / max(1, len(entropies)))
    return generated, decode_ids(tokenizer, generated), mean_entropy, last_state
