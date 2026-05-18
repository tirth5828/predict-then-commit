"""Streaming Transformer backbone with wait-k memory masking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ptc.config import ExperimentConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def waitk_memory_mask(tgt_len: int, src_len: int, k: int, device: torch.device) -> torch.Tensor:
    """Create a wait-k cross-attention mask.

    Target step t may attend only to source positions j <= t+k-1.
    Values are 0 for allowed positions and -inf for blocked positions.
    """
    mask = torch.full((tgt_len, src_len), float("-inf"), device=device)
    for t in range(tgt_len):
        allowed = min(src_len, t + k)
        mask[t, :allowed] = 0.0
    return mask


class StreamingTransformer(nn.Module):
    """Compact seq2seq Transformer used for controlled streaming experiments."""

    def __init__(self, vocab_size: int, pad_id: int, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = cfg.d_model

        self.embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.positional = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(cfg.d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        logits:
            Vocabulary logits of shape [batch, tgt_len, vocab_size].
        decoder_states:
            Decoder hidden states of shape [batch, tgt_len, d_model].
        """
        device = src.device
        src_len = src.size(1)
        tgt_len = tgt.size(1)

        src_mask = torch.zeros((src_len, src_len), dtype=torch.bool, device=device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

        if src_padding_mask is None:
            src_padding_mask = src.eq(self.pad_id)
        if tgt_padding_mask is None:
            tgt_padding_mask = tgt.eq(self.pad_id)

        src_emb = self.positional(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional(self.embedding(tgt) * math.sqrt(self.d_model))

        dec = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.generator(dec), dec


def build_model(vocab_size: int, pad_id: int, cfg: ExperimentConfig) -> StreamingTransformer:
    """Factory for the streaming Transformer."""
    return StreamingTransformer(vocab_size, pad_id, cfg)
