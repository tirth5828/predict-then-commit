"""Dataset loading and tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ptc.config import ExperimentConfig


@dataclass(slots=True)
class ParallelText:
    """Container for parallel source/target strings."""

    sources: List[str]
    targets: List[str]

    def __len__(self) -> int:
        return len(self.sources)


def extract_pair(example: Dict[str, Any], src_lang: str, tgt_lang: str) -> Tuple[str, str]:
    """Extract source/target text from common translation dataset formats."""
    if "translation" in example:
        tr = example["translation"]
        return str(tr[src_lang]), str(tr[tgt_lang])
    if src_lang in example and tgt_lang in example:
        return str(example[src_lang]), str(example[tgt_lang])
    raise KeyError(f"Could not find translation fields for {src_lang}->{tgt_lang}")


def _split_name(split: str, limit: int) -> str:
    return f"{split}[:{limit}]" if limit > 0 else split


def load_parallel_split(cfg: ExperimentConfig, split: str, limit: int) -> ParallelText:
    """Load a parallel-text split, falling back to OPUS Books if WMT is unavailable."""
    try:
        dataset = load_dataset(cfg.primary_dataset, cfg.dataset_config, split=_split_name(split, limit))
    except Exception as first_error:  # pragma: no cover - depends on remote datasets
        try:
            dataset = load_dataset(cfg.fallback_dataset, cfg.fallback_config, split=_split_name(split, limit))
        except Exception as second_error:  # pragma: no cover
            raise RuntimeError(
                "Failed to load both primary and fallback datasets. "
                f"Primary error: {first_error}; fallback error: {second_error}"
            ) from second_error

    srcs, tgts = [], []
    for ex in dataset:
        src, tgt = extract_pair(ex, cfg.src_lang, cfg.tgt_lang)
        srcs.append(src)
        tgts.append(tgt)
    return ParallelText(srcs, tgts)


class TranslationTensorDataset(Dataset):
    """Tokenized parallel text for supervised wait-k training."""

    def __init__(self, data: ParallelText, tokenizer: PreTrainedTokenizerBase, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        enc = tokenizer(
            data.sources,
            text_target=data.targets,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        labels = enc["labels"]
        # Hugging Face convention: -100 labels are ignored by CrossEntropyLoss.
        labels[labels == tokenizer.pad_token_id] = -100
        self.labels = labels

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def make_tokenizer(cfg: ExperimentConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer used for both source and target sides."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_tokenizer)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide pad_token_id.")
    return tokenizer


def make_dataloader(
    data: ParallelText,
    tokenizer: PreTrainedTokenizerBase,
    cfg: ExperimentConfig,
    shuffle: bool,
) -> DataLoader:
    """Build a PyTorch dataloader."""
    dataset = TranslationTensorDataset(data, tokenizer, cfg.max_seq_len)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)
