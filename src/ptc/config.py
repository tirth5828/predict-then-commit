"""Configuration utilities for Predict-Then-Commit experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml


@dataclass(slots=True)
class ExperimentConfig:
    """Typed experiment configuration.

    The values are intentionally explicit so that every reported number can be
    traced back to a single config file. Academic reproducibility is tedious,
    but so are reviewer complaints, and one of those is avoidable.
    """

    run_name: str = "smoke"
    seed: int = 42
    device: str = "auto"

    # Dataset
    primary_dataset: str = "wmt14"
    dataset_config: str = "de-en"
    fallback_dataset: str = "opus_books"
    fallback_config: str = "en-de"
    src_lang: str = "en"
    tgt_lang: str = "de"
    train_samples: int = 128
    valid_samples: int = 64
    test_samples: int = 64
    max_seq_len: int = 48

    # Tokenizer/model
    hf_tokenizer: str = "Helsinki-NLP/opus-mt-en-de"
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 1
    epochs_stage1: int = 1
    epochs_stage2: int = 1
    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-5
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    wait_k: int = 3

    # Reward weights
    lambda_erasure: float = 0.10
    lambda_latency: float = 0.05
    lambda_delay: float = 0.02
    gamma: float = 0.95

    # Evaluation
    waitk_values: List[int] = None  # type: ignore[assignment]
    max_new_tokens: int = 48
    deterministic: bool = True
    use_comet: bool = False
    output_dir: str = "outputs/smoke"

    def __post_init__(self) -> None:
        if self.waitk_values is None:
            self.waitk_values = [1, 3]

    @property
    def torch_device(self) -> torch.device:
        """Resolve the configured device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_REQUIRED_KEYS: set[str] = set()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    missing = _REQUIRED_KEYS - set(data)
    if missing:
        raise KeyError(f"Missing required config keys: {sorted(missing)}")

    return ExperimentConfig(**data)
