#!/usr/bin/env python
"""Evaluate Predict-Then-Commit checkpoints.

Usage:
    python scripts/evaluate.py --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoint.pt
"""

from __future__ import annotations

import argparse

import torch

from ptc.config import load_config
from ptc.data import load_parallel_split, make_tokenizer
from ptc.evaluate import evaluate_systems
from ptc.logging_utils import ensure_dir, save_dataframe, save_jsonl
from ptc.metrics import build_character_mass_tensor
from ptc.model import build_model
from ptc.plotting import save_erasure_barplot, save_pareto_plot
from ptc.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Predict-Then-Commit experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed, deterministic_algorithms=False)
    out = ensure_dir(cfg.output_dir)

    tokenizer = make_tokenizer(cfg)
    test_data = load_parallel_split(cfg, "test", cfg.test_samples)

    model = build_model(len(tokenizer), int(tokenizer.pad_token_id), cfg).to(cfg.torch_device)
    checkpoint = torch.load(args.checkpoint, map_location=cfg.torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    char_mass = build_character_mass_tensor(tokenizer, cfg.torch_device)
    summary, prefix_logs = evaluate_systems(
        model=model,
        tokenizer=tokenizer,
        sources=test_data.sources,
        references=test_data.targets,
        cfg=cfg,
        char_mass=char_mass,
    )

    save_dataframe(summary, out / "summary_metrics.csv")
    save_dataframe(prefix_logs, out / "prefix_logs.csv")
    save_jsonl(prefix_logs.to_dict(orient="records"), out / "prefix_logs.jsonl")
    save_pareto_plot(summary, out / "pareto_frontier.png")
    save_erasure_barplot(summary, out / "erasure_barplot.png")

    print(summary.to_string(index=False))
    print(f"Saved evaluation outputs to {out}")


if __name__ == "__main__":
    main()
