"""Training loops for Predict-Then-Commit."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ptc.config import ExperimentConfig
from ptc.model import StreamingTransformer, waitk_memory_mask
from ptc.policies import CommitPolicy


def train_stage1_mle(
    model: StreamingTransformer,
    loader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Train the translation backbone with a wait-k memory mask."""
    device = cfg.torch_device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage1)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=cfg.label_smoothing)
    rows: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs_stage1):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Stage 1 MLE epoch {epoch + 1}/{cfg.epochs_stage1}")
        for step, batch in enumerate(pbar, start=1):
            src = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tgt_in = labels[:, :-1].clone()
            tgt_out = labels[:, 1:].clone()
            # Replace ignored label positions in decoder input with PAD.
            tgt_in[tgt_in == -100] = int(tokenizer.pad_token_id)

            mem_mask = waitk_memory_mask(tgt_in.size(1), src.size(1), cfg.wait_k, device)
            logits, _ = model(src, tgt_in, memory_mask=mem_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss = loss / cfg.grad_accum_steps
            loss.backward()

            if step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * cfg.grad_accum_steps
            avg = running / step
            pbar.set_postfix(loss=f"{avg:.4f}")
            rows.append({"epoch": epoch + 1, "step": step, "loss": avg})

    return pd.DataFrame(rows)


def train_stage2_policy_placeholder(
    model: StreamingTransformer,
    policy: CommitPolicy,
    loader: DataLoader,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Placeholder policy training loop.

    The production notebook contains the full rollout path. This function keeps
    the package CLI lightweight and honest: it saves the policy initialized for
    evaluation experiments, while avoiding pretending a two-line stub is real RL.
    """
    _ = model, policy, loader, cfg
    return pd.DataFrame([
        {
            "epoch": 0,
            "policy_loss": 0.0,
            "note": "Full policy rollout training is implemented in the notebook; replace this placeholder for long runs.",
        }
    ])
