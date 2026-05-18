#!/usr/bin/env python
"""Train Predict-Then-Commit components.

Usage:
    python scripts/train.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ptc.config import load_config
from ptc.data import load_parallel_split, make_dataloader, make_tokenizer
from ptc.logging_utils import ensure_dir, save_dataframe, save_json
from ptc.model import build_model
from ptc.policies import CommitPolicy
from ptc.reproducibility import set_seed
from ptc.train import train_stage1_mle, train_stage2_policy_placeholder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Predict-Then-Commit experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed, deterministic_algorithms=False)
    out = ensure_dir(cfg.output_dir)

    tokenizer = make_tokenizer(cfg)
    train_data = load_parallel_split(cfg, "train", cfg.train_samples)
    train_loader = make_dataloader(train_data, tokenizer, cfg, shuffle=True)

    model = build_model(len(tokenizer), int(tokenizer.pad_token_id), cfg).to(cfg.torch_device)
    policy = CommitPolicy(input_dim=cfg.d_model + 5).to(cfg.torch_device)

    stage1 = train_stage1_mle(model, train_loader, tokenizer, cfg)
    stage2 = train_stage2_policy_placeholder(model, policy, train_loader, cfg)

    save_dataframe(stage1, out / "train_stage1.csv")
    save_dataframe(stage2, out / "train_stage2.csv")
    save_json(cfg.to_dict(), out / "config.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_state_dict": policy.state_dict(),
            "config": cfg.to_dict(),
        },
        out / "checkpoint.pt",
    )
    print(f"Saved checkpoint to {out / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
