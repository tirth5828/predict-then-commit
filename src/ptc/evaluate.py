"""Evaluation pipeline with true autoregressive baselines and prefix logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ptc.config import ExperimentConfig
from ptc.decoding import add_eos, decode_ids, encode_source_prefix, greedy_decode
from ptc.metrics import cnal_step, corpus_quality_metrics, detokenized_erasure_step
from ptc.model import StreamingTransformer


@dataclass(slots=True)
class SentenceEval:
    prediction: str
    erasure: float
    cnal: float
    delay: float
    logs: List[Dict[str, Any]]


def evaluate_immediate_sentence(
    model: StreamingTransformer,
    tokenizer: PreTrainedTokenizerBase,
    source: str,
    reference: str,
    sent_id: int,
    cfg: ExperimentConfig,
    char_mass: torch.Tensor,
) -> SentenceEval:
    """Immediate retranslation baseline: display full hypothesis at every prefix."""
    src_core = encode_source_prefix(tokenizer, source)
    prev_z = ""
    total_erasure = 0.0
    total_cnal = 0.0
    logs: List[Dict[str, Any]] = []
    final_pred = ""

    for g in range(1, len(src_core) + 1):
        prefix_ids = add_eos(src_core[:g], tokenizer)
        _, h_t, _, _ = greedy_decode(model, tokenizer, prefix_ids, cfg.max_new_tokens)
        z_t = h_t
        step_erasure = detokenized_erasure_step(prev_z, z_t)
        step_cnal, c_src, c_vis = cnal_step(prefix_ids, z_t, char_mass, cfg.gamma)
        total_erasure += step_erasure
        total_cnal += step_cnal
        logs.append(
            {
                "sent_id": sent_id,
                "strategy": "Immediate",
                "prefix_step": g,
                "source_prefix": decode_ids(tokenizer, src_core[:g]),
                "reference": reference,
                "h_t": h_t,
                "c_t": h_t,
                "z_t": z_t,
                "step_erasure": step_erasure,
                "step_cnal": step_cnal,
                "source_char_mass": c_src,
                "visible_char_mass": c_vis,
                "action": "RETRANSLATE",
            }
        )
        prev_z = z_t
        final_pred = z_t

    return SentenceEval(final_pred, total_erasure, total_cnal, 1.0, logs)


def evaluate_waitk_sentence(
    model: StreamingTransformer,
    tokenizer: PreTrainedTokenizerBase,
    source: str,
    reference: str,
    sent_id: int,
    cfg: ExperimentConfig,
    char_mass: torch.Tensor,
    k: int,
) -> SentenceEval:
    """Wait-k baseline with true autoregressive prefix decoding."""
    src_core = encode_source_prefix(tokenizer, source)
    prev_z = ""
    total_erasure = 0.0
    total_cnal = 0.0
    logs: List[Dict[str, Any]] = []
    delays: List[int] = []
    final_pred = ""

    for g in range(1, len(src_core) + 1):
        write_budget = max(0, g - k + 1)
        prefix_ids = add_eos(src_core[:g], tokenizer)
        if write_budget == 0:
            h_t = prev_z
            z_t = prev_z
            action = "READ_ONLY"
        else:
            _, h_t, _, _ = greedy_decode(model, tokenizer, prefix_ids, min(write_budget, cfg.max_new_tokens))
            z_t = h_t
            action = f"WRITE_BUDGET_{write_budget}"
            delays.append(g)

        step_erasure = detokenized_erasure_step(prev_z, z_t)
        step_cnal, c_src, c_vis = cnal_step(prefix_ids, z_t, char_mass, cfg.gamma)
        total_erasure += step_erasure
        total_cnal += step_cnal
        logs.append(
            {
                "sent_id": sent_id,
                "strategy": f"Wait-k(k={k})",
                "prefix_step": g,
                "source_prefix": decode_ids(tokenizer, src_core[:g]),
                "reference": reference,
                "h_t": h_t,
                "c_t": z_t,
                "z_t": z_t,
                "step_erasure": step_erasure,
                "step_cnal": step_cnal,
                "source_char_mass": c_src,
                "visible_char_mass": c_vis,
                "action": action,
            }
        )
        prev_z = z_t
        final_pred = z_t

    # Final flush after complete source context.
    full_ids = add_eos(src_core, tokenizer)
    _, full_hyp, _, _ = greedy_decode(model, tokenizer, full_ids, cfg.max_new_tokens)
    if full_hyp != prev_z:
        step_erasure = detokenized_erasure_step(prev_z, full_hyp)
        step_cnal, c_src, c_vis = cnal_step(full_ids, full_hyp, char_mass, cfg.gamma)
        total_erasure += step_erasure
        total_cnal += step_cnal
        logs.append(
            {
                "sent_id": sent_id,
                "strategy": f"Wait-k(k={k})",
                "prefix_step": len(src_core) + 1,
                "source_prefix": source,
                "reference": reference,
                "h_t": full_hyp,
                "c_t": full_hyp,
                "z_t": full_hyp,
                "step_erasure": step_erasure,
                "step_cnal": step_cnal,
                "source_char_mass": c_src,
                "visible_char_mass": c_vis,
                "action": "FLUSH",
            }
        )
        final_pred = full_hyp

    delay = float(sum(delays) / max(1, len(delays)))
    return SentenceEval(final_pred, total_erasure, total_cnal, delay, logs)


def evaluate_systems(
    model: StreamingTransformer,
    tokenizer: PreTrainedTokenizerBase,
    sources: Sequence[str],
    references: Sequence[str],
    cfg: ExperimentConfig,
    char_mass: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all implemented baselines and return summary + prefix logs."""
    systems: Dict[str, Dict[str, list]] = {"Immediate": {"pred": [], "erasure": [], "cnal": [], "delay": []}}
    for k in cfg.waitk_values:
        systems[f"Wait-k(k={k})"] = {"pred": [], "erasure": [], "cnal": [], "delay": []}

    all_logs: List[Dict[str, Any]] = []
    for sent_id, (src, ref) in enumerate(tqdm(list(zip(sources, references)), desc="Evaluating")):
        res = evaluate_immediate_sentence(model, tokenizer, src, ref, sent_id, cfg, char_mass)
        systems["Immediate"]["pred"].append(res.prediction)
        systems["Immediate"]["erasure"].append(res.erasure)
        systems["Immediate"]["cnal"].append(res.cnal)
        systems["Immediate"]["delay"].append(res.delay)
        all_logs.extend(res.logs)

        for k in cfg.waitk_values:
            name = f"Wait-k(k={k})"
            res = evaluate_waitk_sentence(model, tokenizer, src, ref, sent_id, cfg, char_mass, k)
            systems[name]["pred"].append(res.prediction)
            systems[name]["erasure"].append(res.erasure)
            systems[name]["cnal"].append(res.cnal)
            systems[name]["delay"].append(res.delay)
            all_logs.extend(res.logs)

    rows: List[Dict[str, Any]] = []
    refs = list(references)
    srcs = list(sources)
    for name, values in systems.items():
        quality = corpus_quality_metrics(values["pred"], refs, srcs)
        rows.append(
            {
                "strategy": name,
                **quality,
                "avg_erasure": float(pd.Series(values["erasure"]).mean()),
                "avg_cnal": float(pd.Series(values["cnal"]).mean()),
                "avg_delay": float(pd.Series(values["delay"]).mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(all_logs)
